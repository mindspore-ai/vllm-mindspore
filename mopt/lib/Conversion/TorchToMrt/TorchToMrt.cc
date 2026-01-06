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

#include <algorithm>
#include <iterator>
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
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionOps.h"
#include "mopt/Conversion/MrtTypeConverter.h"
#include "mopt/Conversion/TorchToMrt/TorchAtenToMrt.h"
#include "mopt/Conversion/TorchToMrt/TorchNpuToMrt.h"
#include "mopt/Conversion/TorchToMrt/TorchArithToMrt.h"
#include "mopt/Conversion/TorchToMrt/TorchCustomToMrt.h"
#include "mopt/Dialect/Mrt/Mrt.h"
#include "mopt/Dialect/Mrt/MrtDialect.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"

#include "TorchToMrt.pdll.h.inc"

namespace {

namespace TorchD = mlir::torch::Torch;

// Populate Torch-specific type conversions to MRT types
void populateTorchToMrtTypeConversions(mlir::TypeConverter &converter) {
  converter.addConversion([](TorchD::ValueTensorType type) -> mlir::Type {
    auto optionalSizes = type.getOptionalSizes();
    if (optionalSizes.has_value()) {
      // Normalize dynamic dims: Torch uses -1 / kUnknownSize, MRT uses ShapedType::kDynamic
      llvm::SmallVector<int64_t> shape;
      auto sizes = optionalSizes.value();
      shape.reserve(sizes.size());
      std::transform(sizes.begin(), sizes.end(), std::back_inserter(shape),
                     [](int64_t dim) { return dim < 0 ? mlir::ShapedType::kDynamic : dim; });
      return mrt::TensorType::get(type.getContext(), shape, type.getOptionalDtype(), nullptr);
    } else {
      return mrt::TensorType::get(type.getContext(), std::nullopt, type.getOptionalDtype(), nullptr);
    }
  });

  converter.addConversion([](TorchD::IntType type) -> mlir::Type { return mrt::I64Type::get(type.getContext()); });

  converter.addConversion([](TorchD::FloatType type) -> mlir::Type { return mrt::F64Type::get(type.getContext()); });

  converter.addConversion([](TorchD::BoolType type) -> mlir::Type { return mrt::BooleanType::get(type.getContext()); });

  converter.addConversion(
    [](TorchD::StringType type) -> mlir::Type { return mrt::StringType::get(type.getContext()); });

  converter.addConversion(
    [](TorchD::DeviceType type) -> mlir::Type { return mrt::StringType::get(type.getContext()); });

  converter.addConversion([](TorchD::NoneType type) -> mlir::Type { return mrt::NoneType::get(type.getContext()); });

  converter.addConversion([&](TorchD::ListType type) -> mlir::Type {
    return mrt::ListType::get(type.getContext(), converter.convertType(type.getContainedType()));
  });
}

// TypeConverter for Torch to MRT conversion
class TorchToMrtTypeConverter : public mlir::TypeConverter {
 public:
  TorchToMrtTypeConverter() {
    addConversion([](mlir::Type type) { return type; });
    mrt::populateMrtTypeConversions(*this);
    mrt::populateMrtTypeMaterializations(*this);
    populateTorchToMrtTypeConversions(*this);
  }
};

// Pattern to remove torch_c conversion ops (they become identity/casts in MRT)
template <typename OpTy>
class TorchConversionOpToMrtPattern : public mlir::OpConversionPattern<OpTy> {
 public:
  using mlir::OpConversionPattern<OpTy>::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                                      mlir::ConversionPatternRewriter &rewriter) const override {
    if (adaptor.getOperands().size() != 1 || op->getNumResults() != 1) {
      return mlir::failure();
    }
    rewriter.replaceOp(op, adaptor.getOperand());
    return mlir::success();
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

    // Aten ops
    mlir::populateAtenToMrtConversionPatterns(converter_, patternList);

    // Npu ops
    mlir::populateNpuToMrtConversionPatterns(converter_, patternList);

    // Integer arithmetic ops
    mlir::populateArithToMrtConversionPatterns(converter_, patternList);

    // TorchConversion ops
    patternList.add<TorchConversionOpToMrtPattern<mlir::torch::TorchConversion::ToBuiltinTensorOp>,
                    TorchConversionOpToMrtPattern<mlir::torch::TorchConversion::FromBuiltinTensorOp>>(converter_, ctx);

    // Add a generic pattern to convert any remaining Torch operations to custom_call.
    mlir::populateCustomToMrtConversionPatterns(converter_, patternList);

    patterns_ = std::move(patternList);
    return mlir::success();
  }

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    mlir::MLIRContext *ctx = &getContext();

    mlir::ConversionTarget target(*ctx);
    target.addIllegalDialect<TorchD::TorchDialect>();
    target.addIllegalOp<mlir::torch::TorchConversion::ToBuiltinTensorOp,
                        mlir::torch::TorchConversion::FromBuiltinTensorOp>();
    target.addLegalDialect<mrt::MrtDialect>();
    target.addLegalOp<mlir::UnrealizedConversionCastOp>();
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
