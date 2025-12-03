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

#include "mopt/Conversion/TorchToMrt/TorchToMrt.h"

#include <numeric>

#include "mlir/IR/PatternMatch.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Utils/TorchUpstream.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include "mopt/Conversion/MrtTypeConverter.h"
#include "mopt/Conversion/TorchToMrt/TorchConstantToMrt.h"
#include "mopt/Conversion/TorchToMrt/TorchAtenToMrt.h"
#include "mopt/Dialect/Mrt/Mrt.h"
#include "mopt/Dialect/Mrt/MrtDialect.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"

#include "TorchToMrt.pdll.h.inc"
#include "mopt/Dialect/Mrt/MrtValueBuilder.h"

namespace {

namespace TorchD = mlir::torch::Torch;

// Populate Torch-specific type conversions to MRT types
void populateTorchToMrtTypeConversions(mlir::TypeConverter &converter) {
  converter.addConversion([](TorchD::ValueTensorType type) -> mlir::Type {
    if (auto builtinType = mlir::dyn_cast<mlir::RankedTensorType>(type.toBuiltinTensor())) {
      return mrt::TensorType::get(type.getContext(), builtinType.getShape(), builtinType.getElementType(), nullptr);
    }
    return type;
  });

  converter.addConversion([](TorchD::IntType type) -> mlir::Type { return mrt::I64Type::get(type.getContext()); });

  converter.addConversion([](TorchD::FloatType type) -> mlir::Type { return mrt::F64Type::get(type.getContext()); });

  converter.addConversion([](TorchD::BoolType type) -> mlir::Type { return mrt::BooleanType::get(type.getContext()); });

  converter.addConversion(
    [](TorchD::StringType type) -> mlir::Type { return mrt::StringType::get(type.getContext()); });
}

// TypeConverter for Torch to MRT conversion
class TorchToMrtTypeConverter : public mlir::TypeConverter {
 public:
  TorchToMrtTypeConverter() {
    addConversion([](mlir::Type type) { return type; });
    mrt::populateMrtTypeConversions(*this);
    populateTorchToMrtTypeConversions(*this);
  }
};

//===----------------------------------------------------------------------===//
// Pass definition
//===----------------------------------------------------------------------===//

struct ConvertTorchToMRTPass : public mlir::PassWrapper<ConvertTorchToMRTPass, mlir::OperationPass<mlir::ModuleOp>> {
  mlir::StringRef getArgument() const final { return "convert-torch-to-mrt"; }

  mlir::StringRef getDescription() const final { return "Convert Torch operations to MRT dialect operations"; }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<TorchD::TorchDialect>();
    registry.insert<mrt::MrtDialect>();
    registry.insert<mlir::func::FuncDialect>();
    registry.insert<mlir::pdl::PDLDialect>();
    registry.insert<mlir::pdl_interp::PDLInterpDialect>();
  }

  mlir::LogicalResult initialize(mlir::MLIRContext *ctx) override {
    mlir::RewritePatternSet patternList(ctx);
    mlir::registerConversionPDLFunctions(patternList);
    populateGeneratedPDLLPatterns(patternList, mlir::PDLConversionConfig(&converter_));
    mlir::populateFunctionOpInterfaceTypeConversionPattern<mlir::func::FuncOp>(patternList, converter_);

    // Torch constant patterns
    mlir::populateTorchConstantToMrtPatterns(converter_, patternList);
    // Aten ops
    mlir::populateAtenToMrtConversionPatterns(converter_, patternList);

    patterns_ = std::move(patternList);
    return mlir::success();
  }

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    mlir::MLIRContext *ctx = &getContext();

    mlir::ConversionTarget target(*ctx);
    target.addIllegalDialect<TorchD::TorchDialect>();
    // torch.constant.none is legal (unused constants will be eliminated by DCE)
    target.addLegalOp<TorchD::ConstantNoneOp>();
    target.addLegalDialect<mrt::MrtDialect>();
    target.addDynamicallyLegalOp<mlir::func::FuncOp>(
      [&](mlir::func::FuncOp op) { return converter_.isSignatureLegal(op.getFunctionType()); });
    target.addDynamicallyLegalOp<mlir::func::ReturnOp>(
      [&](mlir::func::ReturnOp op) { return converter_.isLegal(op.getOperandTypes()); });

    if (mlir::failed(mlir::applyPartialConversion(module, target, patterns_))) {
      signalPassFailure();
    }
  }

  mlir::FrozenRewritePatternSet patterns_;
  TorchToMrtTypeConverter converter_;
};
}  // namespace

namespace mlir {
std::unique_ptr<Pass> createConvertTorchToMRTPass() { return std::make_unique<ConvertTorchToMRTPass>(); }
}  // namespace mlir
