/**
 * Copyright 2025 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "mopt/Conversion/TorchToMrt/TorchAtenToMrt.h"

#include <numeric>
#include <string>

#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/DialectConversion.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Utils/TorchUpstream.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include "mopt/Dialect/Mrt/Mrt.h"
#include "mopt/Dialect/Mrt/MrtValueBuilder.h"

namespace mlir {

namespace TorchD = mlir::torch::Torch;

//===----------------------------------------------------------------------===//
// Custom conversion patterns
//===----------------------------------------------------------------------===//

struct ConvertAtenTransposeInt : public OpConversionPattern<TorchD::AtenTransposeIntOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(TorchD::AtenTransposeIntOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value self = adaptor.getSelf();
    int64_t dim0, dim1;

    if (!matchPattern(op.getDim0(), TorchD::m_TorchConstantInt(&dim0)))
      return rewriter.notifyMatchFailure(op, "dim0 must be constant");
    if (!matchPattern(op.getDim1(), TorchD::m_TorchConstantInt(&dim1)))
      return rewriter.notifyMatchFailure(op, "dim1 must be constant");

    auto inType = cast<mrt::TensorType>(self.getType());
    int64_t inputRank = inType.getRank();
    auto outType = cast<mrt::TensorType>(getTypeConverter()->convertType(op->getResult(0).getType()));

    dim0 = TorchD::toPositiveDim(dim0, inputRank);
    if (!TorchD::isValidDim(dim0, inputRank)) return rewriter.notifyMatchFailure(op, "dim0 out of range");

    dim1 = TorchD::toPositiveDim(dim1, inputRank);
    if (!TorchD::isValidDim(dim1, inputRank)) return rewriter.notifyMatchFailure(op, "dim1 out of range");

    llvm::SmallVector<int64_t, 4> permValues(inputRank);
    std::iota(std::begin(permValues), std::end(permValues), 0);
    std::swap(permValues[dim0], permValues[dim1]);

    auto permValue = mrt::MrtValueBuilder(rewriter).createI64Array(op.getLoc(), permValues);
    rewriter.replaceOpWithNewOp<mrt::PermuteOp>(op, outType, self, permValue);
    return success();
  }
};

struct ConvertAtenToDtype : public OpConversionPattern<TorchD::AtenToDtypeOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(TorchD::AtenToDtypeOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value self = adaptor.getSelf();
    auto outType = cast<mrt::TensorType>(getTypeConverter()->convertType(op.getType()));

    auto dtypeValue = mrt::MrtValueBuilder(rewriter).createDtype(op.getLoc(), outType.getElementType());
    rewriter.replaceOpWithNewOp<mrt::CastOp>(op, outType, self, dtypeValue);
    return success();
  }
};

struct ConvertAtenDivTensorMode : public OpConversionPattern<TorchD::AtenDivTensorModeOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(TorchD::AtenDivTensorModeOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto outType = cast<mrt::TensorType>(getTypeConverter()->convertType(op.getType()));

    std::string roundingMode;
    if (!matchPattern(op.getRoundingMode(), TorchD::m_TorchConstantStr(roundingMode)))
      return rewriter.notifyMatchFailure(op, "rounding_mode must be a constant string");

    int64_t mode = 0;
    if (roundingMode == "floor") {
      mode = 2;
    } else if (roundingMode == "trunc") {
      mode = 1;
    }

    auto modeValue = mrt::MrtValueBuilder(rewriter).createI64(op.getLoc(), mode);
    rewriter.replaceOpWithNewOp<mrt::DivModOp>(op, outType, adaptor.getSelf(), adaptor.getOther(), modeValue);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pattern population
//===----------------------------------------------------------------------===//

// Populate custom (hand-written) Aten ops to MRT conversion patterns
static void populateAtenToMrtCustomPatterns(TypeConverter &converter, RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();

  patterns.add<ConvertAtenDivTensorMode>(converter, context);
  patterns.add<ConvertAtenToDtype>(converter, context);
  patterns.add<ConvertAtenTransposeInt>(converter, context);
}

// Populate all Aten ops to MRT conversion patterns
void populateAtenToMrtConversionPatterns(TypeConverter &converter, RewritePatternSet &patterns) {
  populateAtenToMrtCustomPatterns(converter, patterns);
}

}  // namespace mlir
