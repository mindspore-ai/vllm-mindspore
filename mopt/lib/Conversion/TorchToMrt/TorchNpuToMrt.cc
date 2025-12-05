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

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "mopt/Dialect/Mrt/Mrt.h"

namespace mlir {

namespace TorchD = mlir::torch::Torch;

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

void populateNpuToMrtConversionPatterns(TypeConverter &converter, RewritePatternSet &patterns) {
  patterns.add<ConvertNpuApplyRotaryPosEmb>(converter, patterns.getContext());
}

}  // namespace mlir
