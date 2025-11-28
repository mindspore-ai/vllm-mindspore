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

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "mopt/Dialect/Mrt/Mrt.h"
#include "mopt/Dialect/Mrt/MrtDialect.h"
#include "mlir/IR/BuiltinOps.h"

namespace {

class TorchToMrtTypeConverter : public mlir::TypeConverter {
 public:
  explicit TorchToMrtTypeConverter(mlir::MLIRContext *ctx) {
    addConversion([](mlir::Type type) { return type; });  // Default identity

    addConversion([ctx](mlir::torch::Torch::ValueTensorType type) -> mlir::Type {
      if (auto builtinType = mlir::dyn_cast<mlir::RankedTensorType>(type.toBuiltinTensor())) {
        return mrt::TensorType::get(ctx, builtinType.getShape(), builtinType.getElementType(), nullptr);
      }
      return type;
    });

    addConversion([ctx](mlir::RankedTensorType type) -> mlir::Type {
      return mrt::TensorType::get(ctx, type.getShape(), type.getElementType(), nullptr);
    });

    addConversion([ctx](mlir::torch::Torch::IntType type) -> mlir::Type { return mrt::I64Type::get(ctx); });
  }
};

struct ConvertSymbolicInt : public mlir::OpConversionPattern<mlir::torch::Torch::SymbolicIntOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(mlir::torch::Torch::SymbolicIntOp op, OpAdaptor adaptor,
                                      mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Type resultType = getTypeConverter()->convertType(op.getType());
    if (!resultType) return mlir::failure();

    rewriter.replaceOpWithNewOp<mrt::SymbolicIntOp>(op, resultType, op.getSymbolName(), op.getMinVal(), op.getMaxVal());
    return mlir::success();
  }
};

struct ConvertBindSymbolicShape : public mlir::OpConversionPattern<mlir::torch::Torch::BindSymbolicShapeOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(mlir::torch::Torch::BindSymbolicShapeOp op, OpAdaptor adaptor,
                                      mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mrt::BindSymbolicShapeOp>(op, adaptor.getOperand(), adaptor.getShapeSymbols(),
                                                          op.getShapeExpressions());
    return mlir::success();
  }
};

struct ConvertAtenMulTensor : public mlir::OpConversionPattern<mlir::torch::Torch::AtenMulTensorOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(mlir::torch::Torch::AtenMulTensorOp op, OpAdaptor adaptor,
                                      mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Type resultType = getTypeConverter()->convertType(op.getType());
    if (!resultType) return mlir::failure();

    rewriter.replaceOpWithNewOp<mrt::MulOp>(op, resultType, adaptor.getSelf(), adaptor.getOther());
    return mlir::success();
  }
};

struct ConvertReturnOp : public mlir::OpConversionPattern<mlir::func::ReturnOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(mlir::func::ReturnOp op, OpAdaptor adaptor,
                                      mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::func::ReturnOp>(op, adaptor.getOperands());
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
    registry.insert<mlir::torch::Torch::TorchDialect>();
    registry.insert<mrt::MrtDialect>();
    registry.insert<mlir::func::FuncDialect>();
  }

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    mlir::MLIRContext *context = &getContext();

    TorchToMrtTypeConverter converter(context);
    mlir::RewritePatternSet patterns(context);

    patterns.add<ConvertSymbolicInt, ConvertBindSymbolicShape>(converter, context);
    patterns.add<ConvertAtenMulTensor>(converter, context);
    patterns.add<ConvertReturnOp>(converter, context);
    mlir::populateFunctionOpInterfaceTypeConversionPattern<mlir::func::FuncOp>(patterns, converter);

    mlir::ConversionTarget target(*context);
    target.addLegalDialect<mrt::MrtDialect>();
    target.addLegalOp<mlir::UnrealizedConversionCastOp>();

    target.addDynamicallyLegalOp<mlir::func::FuncOp>(
      [&](mlir::func::FuncOp op) { return converter.isSignatureLegal(op.getFunctionType()); });

    target.addDynamicallyLegalOp<mlir::func::ReturnOp>(
      [&](mlir::func::ReturnOp op) { return converter.isLegal(op.getOperandTypes()); });

    target.addIllegalOp<mlir::torch::Torch::SymbolicIntOp>();
    target.addIllegalOp<mlir::torch::Torch::BindSymbolicShapeOp>();
    target.addIllegalOp<mlir::torch::Torch::AtenMulTensorOp>();

    if (mlir::failed(mlir::applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
}  // namespace

namespace mlir {
std::unique_ptr<Pass> createConvertTorchToMRTPass() { return std::make_unique<ConvertTorchToMRTPass>(); }
}  // namespace mlir
