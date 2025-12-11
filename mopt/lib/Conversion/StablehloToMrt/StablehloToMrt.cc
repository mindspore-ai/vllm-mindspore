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

#include "mopt/Conversion/StablehloToMrt/StablehloToMrt.h"
#include "mopt/Conversion/ConversionPatternTemplates.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/SmallVector.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "mopt/Conversion/MrtTypeConverter.h"
#include "mopt/Dialect/Mrt/Mrt.h"
#include "mopt/Dialect/Mrt/MrtDialect.h"

using llvm::APFloat;
using mlir::ArrayRef;
using mlir::cast;
using mlir::ConversionPattern;
using mlir::ConversionPatternRewriter;
using mlir::ConversionTarget;
using mlir::dyn_cast;
using mlir::failed;
using mlir::failure;
using mlir::isa;
using mlir::LogicalResult;
using mlir::MLIRContext;
using mlir::ModuleOp;
using mlir::OpConversionPattern;
using mlir::Operation;
using mlir::OperationPass;
using mlir::Pass;
using mlir::PassWrapper;
using mlir::RankedTensorType;
using mlir::RewritePatternSet;
using mlir::ShapedType;
using mlir::SmallVector;
using mlir::StringRef;
using mlir::success;
using mlir::Type;
using mlir::TypeAttr;
using mlir::TypeConverter;
using mlir::Value;
using mlir::ValueRange;

namespace {

// TypeConverter for StableHLO to MRT conversion
class StablehloToMrtTypeConverter : public mlir::TypeConverter {
 public:
  explicit StablehloToMrtTypeConverter(mlir::MLIRContext *ctx) {
    addConversion([](mlir::Type type) { return type; });
    mrt::populateMrtTypeConversions(*this);
  }
};

//===----------------------------------------------------------------------===//
// Conversion patterns
//===----------------------------------------------------------------------===//

// Convert stablehlo.reshape to mrt.reshape
struct ConvertReshapeOp : public OpConversionPattern<mlir::stablehlo::ReshapeOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::ReshapeOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type resultType = getTypeConverter()->convertType(op.getResult().getType());
    auto shapedType = dyn_cast<ShapedType>(resultType);
    if (!shapedType || !shapedType.hasRank()) {
      return failure();
    }

    auto shapeValue = rewriter.create<mrt::CreateI64ArrayOp>(op->getLoc(), shapedType.getShape());
    rewriter.replaceOpWithNewOp<mrt::ReshapeOp>(op, resultType, adaptor.getOperand(), shapeValue);
    return success();
  }
};

// Convert stablehlo.convert to mrt.cast
struct ConvertConvertOp : public OpConversionPattern<mlir::stablehlo::ConvertOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::ConvertOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type resultType = getTypeConverter()->convertType(op.getResult().getType());
    auto shapedType = dyn_cast<ShapedType>(resultType);
    if (!shapedType || !shapedType.hasRank()) {
      return failure();
    }

    auto dtypeValue = rewriter.create<mrt::CreateDtypeOp>(op->getLoc(), shapedType.getElementType());
    rewriter.replaceOpWithNewOp<mrt::CastOp>(op, resultType, adaptor.getOperand(), dtypeValue);
    return success();
  }
};

// Convert stablehlo.divide to mrt.div
struct ConvertDivideOp : public OpConversionPattern<mlir::stablehlo::DivOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::DivOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type resultType = getTypeConverter()->convertType(op.getResult().getType());
    rewriter.replaceOpWithNewOp<mrt::DivOp>(op, resultType, adaptor.getLhs(), adaptor.getRhs());
    return success();
  }
};

// Convert stablehlo.remainder to mrt.remainder_tensor_tensor
struct ConvertRemainderOp : public OpConversionPattern<mlir::stablehlo::RemOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::RemOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type resultType = getTypeConverter()->convertType(op.getResult().getType());
    rewriter.replaceOpWithNewOp<mrt::RemainderTensorTensorOp>(op, resultType, adaptor.getLhs(), adaptor.getRhs());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass definition
//===----------------------------------------------------------------------===//

struct ConvertStablehloToMRTPass : public PassWrapper<ConvertStablehloToMRTPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertStablehloToMRTPass)

  StringRef getArgument() const final { return "convert-stablehlo-to-mrt"; }

  StringRef getDescription() const final { return "Convert StableHLO operations to MRT dialect operations"; }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::stablehlo::StablehloDialect>();
    registry.insert<mrt::MrtDialect>();
    registry.insert<mlir::arith::ArithDialect>();
    registry.insert<mlir::func::FuncDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();

    StablehloToMrtTypeConverter typeConverter(ctx);

    ConversionTarget target(*ctx);
    target.addLegalDialect<mrt::MrtDialect>();
    target.addLegalDialect<mlir::arith::ArithDialect>();
    target.addLegalOp<mlir::ModuleOp>();

    // Mark func.func as dynamically legal (need type conversion)
    target.addDynamicallyLegalOp<mlir::func::FuncOp>(
      [&](mlir::func::FuncOp op) { return typeConverter.isSignatureLegal(op.getFunctionType()); });
    target.addDynamicallyLegalOp<mlir::func::ReturnOp>(
      [&](mlir::func::ReturnOp op) { return typeConverter.isLegal(op.getOperandTypes()); });

    target.addIllegalDialect<mlir::stablehlo::StablehloDialect>();

    RewritePatternSet patterns(ctx);

    // Add simple binary op conversions using MOPT_CONVERT
    patterns.add<MOPT_CONVERT(mlir::stablehlo::MulOp, mrt::MulOp, Lhs, Rhs)>(typeConverter, ctx);
    patterns.add<MOPT_CONVERT(mlir::stablehlo::DivOp, mrt::DivOp, Lhs, Rhs)>(typeConverter, ctx);
    patterns.add<MOPT_CONVERT(mlir::stablehlo::RemOp, mrt::RemainderTensorTensorOp, Lhs, Rhs)>(typeConverter, ctx);

    // Add complex patterns
    patterns.add<ConvertReshapeOp>(typeConverter, ctx);
    patterns.add<ConvertConvertOp>(typeConverter, ctx);

    // Add function signature conversion patterns
    mlir::populateFunctionOpInterfaceTypeConversionPattern<mlir::func::FuncOp>(patterns, typeConverter);
    mlir::populateReturnOpTypeConversionPattern(patterns, typeConverter);

    if (failed(mlir::applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

namespace mlir {

std::unique_ptr<Pass> createConvertStablehloToMRTPass() { return std::make_unique<ConvertStablehloToMRTPass>(); }

}  // namespace mlir
