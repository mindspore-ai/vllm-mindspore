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

#include "mopt/Conversion/TorchToMrt/TorchConstantToMrt.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "mopt/Dialect/Mrt/Mrt.h"

namespace {

namespace TorchD = mlir::torch::Torch;

//===----------------------------------------------------------------------===//
// Constant conversion patterns
//===----------------------------------------------------------------------===//

struct ConvertConstantInt : public mlir::OpConversionPattern<TorchD::ConstantIntOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(TorchD::ConstantIntOp op, OpAdaptor adaptor,
                                      mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Type resultType = getTypeConverter()->convertType(op.getType());
    if (!resultType) return mlir::failure();

    int64_t value = op.getValue();
    auto valueAttr = rewriter.getI64IntegerAttr(value);
    rewriter.replaceOpWithNewOp<mrt::CreateI64Op>(op, resultType, valueAttr);
    return mlir::success();
  }
};

struct ConvertConstantFloat : public mlir::OpConversionPattern<TorchD::ConstantFloatOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(TorchD::ConstantFloatOp op, OpAdaptor adaptor,
                                      mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Type resultType = getTypeConverter()->convertType(op.getType());
    if (!resultType) return mlir::failure();

    double value = op.getValue().convertToDouble();
    auto valueAttr = rewriter.getF64FloatAttr(value);
    rewriter.replaceOpWithNewOp<mrt::CreateF64Op>(op, resultType, valueAttr);
    return mlir::success();
  }
};

struct ConvertConstantBool : public mlir::OpConversionPattern<TorchD::ConstantBoolOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(TorchD::ConstantBoolOp op, OpAdaptor adaptor,
                                      mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Type resultType = getTypeConverter()->convertType(op.getType());
    if (!resultType) return mlir::failure();

    bool value = op.getValue();
    auto valueAttr = rewriter.getBoolAttr(value);
    rewriter.replaceOpWithNewOp<mrt::CreateBooleanOp>(op, resultType, valueAttr);
    return mlir::success();
  }
};

struct ConvertConstantStr : public mlir::OpConversionPattern<TorchD::ConstantStrOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(TorchD::ConstantStrOp op, OpAdaptor adaptor,
                                      mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Type resultType = getTypeConverter()->convertType(op.getType());
    if (!resultType) return mlir::failure();

    llvm::StringRef value = op.getValue();
    auto valueAttr = rewriter.getStringAttr(value);
    rewriter.replaceOpWithNewOp<mrt::CreateStringOp>(op, resultType, valueAttr);
    return mlir::success();
  }
};

}  // namespace

namespace mlir {

void populateTorchConstantToMrtPatterns(TypeConverter &converter, RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();
  patterns.add<ConvertConstantInt, ConvertConstantFloat, ConvertConstantBool, ConvertConstantStr>(converter, context);
}

}  // namespace mlir
