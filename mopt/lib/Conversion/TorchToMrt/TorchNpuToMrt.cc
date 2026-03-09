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

#include "mopt/Conversion/TorchToMrt/TorchNpuToMrt.h"

#include <cmath>
#include <cstddef>
#include <optional>

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "mopt/Dialect/Mrt/Mrt.h"

namespace mlir {

namespace TorchD = mlir::torch::Torch;

namespace {
std::optional<int64_t> GetI64ConstValue(Value value) {
  if (auto i64Const = value.getDefiningOp<mrt::CreateI64Op>()) {
    return i64Const.getValueAttr().getInt();
  }
  return std::nullopt;
}

std::optional<double> GetF64ConstValue(Value value) {
  if (auto f64Const = value.getDefiningOp<mrt::CreateF64Op>()) {
    return f64Const.getValueAttr().getValueAsDouble();
  }
  return std::nullopt;
}

bool IsNearlyEqual(double lhs, double rhs, double tol = 1e-6) { return std::abs(lhs - rhs) <= tol; }
}  // namespace

struct ConvertNpuApplyRotaryPosEmb : public OpConversionPattern<TorchD::OperatorOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(TorchD::OperatorOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    if (op.getName() != "torch.npu.npu_apply_rotary_pos_emb") return failure();

    auto operands = adaptor.getOperands();
    if (operands.size() < 4) return rewriter.notifyMatchFailure(op, "expected at least 4 operands");

    auto one = rewriter.create<mrt::CreateI64Op>(op.getLoc(), static_cast<int64_t>(1));

    SmallVector<Type, 4> resultTypes;
    if (failed(getTypeConverter()->convertTypes(op.getResultTypes(), resultTypes)))
      return rewriter.notifyMatchFailure(op, "type conversion failed");

    rewriter.replaceOpWithNewOp<mrt::ApplyRotaryPosEmbOp>(op, resultTypes, operands[0], operands[1], operands[2],
                                                          operands[3], one);

    return success();
  }
};

struct ConvertNpuDequantSwigluQuant : public OpConversionPattern<TorchD::OperatorOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(TorchD::OperatorOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    if (op.getName() != "torch.npu.npu_dequant_swiglu_quant") return failure();

    auto operands = adaptor.getOperands();
    constexpr size_t kExpectedMinInputSize = 13;
    if (operands.size() < kExpectedMinInputSize) {
      return rewriter.notifyMatchFailure(op, "expected 13 operands for npu_dequant_swiglu_quant");
    }

    // InferRT uses aclnnDequantSwigluQuant (V1). The following V2-only controls
    // must keep defaults; otherwise fallback to generic custom_call path.
    auto quantModeOpt = GetI64ConstValue(operands[8]);
    auto swigluModeOpt = GetI64ConstValue(operands[9]);
    auto clampLimitOpt = GetF64ConstValue(operands[10]);
    auto gluAlphaOpt = GetF64ConstValue(operands[11]);
    auto gluBiasOpt = GetF64ConstValue(operands[12]);
    if (!quantModeOpt || !swigluModeOpt || !clampLimitOpt || !gluAlphaOpt || !gluBiasOpt) {
      return failure();
    }
    if ((*quantModeOpt != 0 && *quantModeOpt != 1) || *swigluModeOpt != 0 || !IsNearlyEqual(*clampLimitOpt, 7.0) ||
        !IsNearlyEqual(*gluAlphaOpt, 1.702) || !IsNearlyEqual(*gluBiasOpt, 1.0)) {
      return failure();
    }

    auto quantModeValue = rewriter.create<mrt::CreateStringOp>(
      op.getLoc(), rewriter.getStringAttr(*quantModeOpt == 0 ? "static" : "dynamic"));

    SmallVector<Type, 4> resultTypes;
    if (failed(getTypeConverter()->convertTypes(op.getResultTypes(), resultTypes)))
      return rewriter.notifyMatchFailure(op, "type conversion failed");

    rewriter.replaceOpWithNewOp<mrt::DequantSwigluQuantOp>(op, resultTypes, operands[0], operands[1], operands[2],
                                                           operands[3], operands[4], operands[5], operands[6],
                                                           operands[7], quantModeValue);

    return success();
  }
};

void populateNpuToMrtConversionPatterns(TypeConverter &converter, RewritePatternSet &patterns) {
  patterns.add<ConvertNpuApplyRotaryPosEmb, ConvertNpuDequantSwigluQuant>(converter, patterns.getContext());
}

}  // namespace mlir
