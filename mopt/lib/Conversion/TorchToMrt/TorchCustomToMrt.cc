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

#include "mopt/Conversion/TorchToMrt/TorchCustomToMrt.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Transforms/DialectConversion.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "mopt/Dialect/Mrt/Mrt.h"
#include "mopt/Conversion/CommonToMrt/DvmSerialization.h"

namespace mlir {
namespace TorchD = mlir::torch::Torch;

struct ConvertMfusionDvmCall : public OpConversionPattern<TorchD::OperatorOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(TorchD::OperatorOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    std::string fullOpName = op.getNameAttr().str();
    if (fullOpName.find("torch.mfusion.dvm_call") == std::string::npos) {
      return failure();
    }

    auto subgraphMlirAttr = op->getAttrOfType<StringAttr>("mfusion.subgraph_mlir");
    if (!subgraphMlirAttr) {
      return op->emitError("missing 'mfusion.subgraph_mlir' attribute on mfusion dvm_call");
    }

    auto isDynamicAttr = op->getAttrOfType<BoolAttr>("mfusion.is_dynamic");
    if (!isDynamicAttr) {
      return op->emitError("missing 'mfusion.is_dynamic' attribute on mfusion dvm_call");
    }

    // Parse the DVM MLIR string back into a module
    auto dvmModule = parseSourceString<ModuleOp>(subgraphMlirAttr.getValue(), op->getContext());
    if (!dvmModule) {
      return op->emitError("failed to parse 'mfusion.subgraph_mlir' attribute");
    }

    auto funcOps = dvmModule->getOps<func::FuncOp>();
    if (funcOps.empty()) {
      return op->emitError("no func.func found in mfusion dvm_call subgraph");
    }
    auto funcOp = *funcOps.begin();

    std::string payloadJson;
    llvm::SmallVector<Value, 8> dvmInputs;
    std::string kernelType = isDynamicAttr.getValue() ? "dyn_shape" : "static_shape";
    if (failed(dvm::serializeDvmFuncToJson(funcOp, kernelType, 0, payloadJson, dvmInputs))) {
      return op->emitError("failed to serialize DVM function to JSON");
    }

    auto operands = adaptor.getOperands();
    // The last operand of mfusion torch.operator is the subgraph name string constant
    if (operands.empty()) {
      return op->emitError("mfusion dvm_call expects at least one operand (subgraph name)");
    }
    auto tensorInputs = operands.drop_back();

    auto payloadAttr = StringAttr::get(op->getContext(), payloadJson);
    auto payloadVal =
      rewriter.create<mrt::CreateStringOp>(op->getLoc(), mrt::StringType::get(op->getContext()), payloadAttr);

    SmallVector<Type, 4> resultTypes;
    if (failed(getTypeConverter()->convertTypes(op.getResultTypes(), resultTypes)))
      return rewriter.notifyMatchFailure(op, "type conversion failed");

    rewriter.replaceOpWithNewOp<mrt::DvmCallOp>(op, resultTypes, payloadVal, tensorInputs);
    return success();
  }
};

struct ConvertCustomOpToCustomCall : public OpConversionPattern<TorchD::OperatorOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(TorchD::OperatorOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto operands = adaptor.getOperands();
    std::string fullOpName = op.getNameAttr().str();
    auto opName = rewriter.create<mrt::CreateStringOp>(op->getLoc(), rewriter.getStringAttr(fullOpName));

    // Prepare operands for CustomCall (first operand is the op name)
    SmallVector<Value, 4> operandsForCustomCall;
    operandsForCustomCall.push_back(opName.getResult());
    operandsForCustomCall.append(operands.begin(), operands.end());
    SmallVector<Type, 4> resultTypes;
    if (failed(getTypeConverter()->convertTypes(op.getResultTypes(), resultTypes)))
      return rewriter.notifyMatchFailure(op, "type conversion failed");

    rewriter.replaceOpWithNewOp<mrt::CustomCallOp>(op, resultTypes, operandsForCustomCall);
    return success();
  }
};

class ConvertTorchOpToCustomCall : public mlir::ConversionPattern {
 public:
  explicit ConvertTorchOpToCustomCall(TypeConverter &converter, MLIRContext *ctx)
      : ConversionPattern(converter, Pattern::MatchAnyOpTypeTag(), /*benefit=*/0, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                                ConversionPatternRewriter &rewriter) const override {
    // Only convert Torch dialect operations
    if (!op->getDialect() || !llvm::isa<TorchD::TorchDialect>(op->getDialect())) {
      return failure();
    }

    std::string fullOpName = op->getName().getStringRef().str();
    auto opName = rewriter.create<mrt::CreateStringOp>(op->getLoc(), rewriter.getStringAttr(fullOpName));

    // Prepare operands for CustomCall (first operand is the op name)
    SmallVector<Value, 4> operandsForCustomCall;
    operandsForCustomCall.push_back(opName.getResult());
    operandsForCustomCall.append(operands.begin(), operands.end());

    // Convert the result types
    SmallVector<Type, 4> resultTypes;
    if (failed(getTypeConverter()->convertTypes(op->getResultTypes(), resultTypes))) {
      return failure();
    }

    rewriter.replaceOpWithNewOp<mrt::CustomCallOp>(op, resultTypes, operandsForCustomCall);
    return success();
  }
};

// This pass should be called last, handling all unprocessed torch dialect operators as custom_call
void populateCustomToMrtConversionPatterns(TypeConverter &converter, RewritePatternSet &patterns) {
  patterns.add<ConvertMfusionDvmCall>(converter, patterns.getContext(), /*benefit=*/10);
  patterns.add<ConvertCustomOpToCustomCall>(converter, patterns.getContext(), /*benefit=*/0);
  patterns.add<ConvertTorchOpToCustomCall>(converter, patterns.getContext());
}

}  // namespace mlir
