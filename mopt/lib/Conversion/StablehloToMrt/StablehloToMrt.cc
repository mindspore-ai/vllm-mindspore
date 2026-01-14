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
#include "mopt/Conversion/StablehloToMrt/StablehloToMrtTypeConverter.h"
#include "mopt/Conversion/ConversionPatternTemplates.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
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
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionDialect.h"
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

static mlir::Value CreateAlphaOneForElementType(mlir::Location loc, mlir::Type elementType,
                                                mlir::ConversionPatternRewriter &rewriter) {
  // MRT add/sub take an "alpha" scalar: result = x (+|-) y * alpha.
  // Use alpha=1 with a scalar type compatible with the tensor element type.
  // - For float-like element types (f16/f32/f64/bf16), use f64(1.0).
  // - For integer/bool element types, use i64(1).
  if (mlir::isa<mlir::FloatType>(elementType) || mlir::isa<mlir::BFloat16Type>(elementType)) {
    auto one = rewriter.getF64FloatAttr(1.0);
    return rewriter.create<mrt::CreateF64Op>(loc, mrt::F64Type::get(rewriter.getContext()), one).getResult();
  }
  auto one = rewriter.getI64IntegerAttr(1);
  return rewriter.create<mrt::CreateI64Op>(loc, mrt::I64Type::get(rewriter.getContext()), one).getResult();
}

// Convert stablehlo.add to mrt.add (alpha=1)
struct ConvertAddOp : public OpConversionPattern<mlir::stablehlo::AddOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::AddOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type resultType = getTypeConverter()->convertType(op.getResult().getType());
    // NOTE: resultType is typically converted to !mrt.tensor<...>, which is NOT
    // a builtin mlir::ShapedType. So we must query element type from the
    // original StableHLO tensor type instead of the converted result type.
    auto originalTensorTy = dyn_cast<mlir::TensorType>(op.getType());
    if (!originalTensorTy) return failure();
    auto alpha = CreateAlphaOneForElementType(op.getLoc(), originalTensorTy.getElementType(), rewriter);
    rewriter.replaceOpWithNewOp<mrt::AddOp>(op, resultType, adaptor.getLhs(), adaptor.getRhs(), alpha);
    return success();
  }
};

// Convert stablehlo.subtract to mrt.sub (alpha=1)
struct ConvertSubtractOp : public OpConversionPattern<mlir::stablehlo::SubtractOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::SubtractOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type resultType = getTypeConverter()->convertType(op.getResult().getType());
    auto originalTensorTy = dyn_cast<mlir::TensorType>(op.getType());
    if (!originalTensorTy) return failure();
    auto alpha = CreateAlphaOneForElementType(op.getLoc(), originalTensorTy.getElementType(), rewriter);
    rewriter.replaceOpWithNewOp<mrt::SubOp>(op, resultType, adaptor.getLhs(), adaptor.getRhs(), alpha);
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

// Convert stablehlo.dot to mrt.mm/mrt.matmul
//
// StableHLO DotOp covers vector/matrix dot products. In practice for LLM graphs
// we most commonly see 2D matmul. MRT provides:
// - mrt.mm: 2D x 2D -> 2D
// - mrt.matmul: 1D..6D general matmul
struct ConvertDotOp : public OpConversionPattern<mlir::stablehlo::DotOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::DotOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type resultType = getTypeConverter()->convertType(op.getResult().getType());

    auto lhsTy = dyn_cast<RankedTensorType>(op.getLhs().getType());
    auto rhsTy = dyn_cast<RankedTensorType>(op.getRhs().getType());
    auto resTy = dyn_cast<RankedTensorType>(op.getResult().getType());
    if (!lhsTy || !rhsTy || !resTy) {
      return failure();
    }

    // Prefer mrt.mm for the common 2D case; fall back to mrt.matmul otherwise.
    if (lhsTy.getRank() == 2 && rhsTy.getRank() == 2 && resTy.getRank() == 2) {
      rewriter.replaceOpWithNewOp<mrt::Mm>(op, resultType, adaptor.getLhs(), adaptor.getRhs());
      return success();
    }

    rewriter.replaceOpWithNewOp<mrt::Matmul>(op, resultType, adaptor.getLhs(), adaptor.getRhs());
    return success();
  }
};

// Convert stablehlo.dynamic_broadcast_in_dim to mrt.expand when the result shape
// is representable as a size array (static ints + -1 for dynamic dims).
//
// NOTE: MRT currently models expand sizes as an i64 array value (typically a
// constant list). We conservatively lower only the common "same-rank, identity
// broadcast_dimensions" case which arises from elementwise broadcasting.
struct ConvertDynamicBroadcastInDimOp : public OpConversionPattern<mlir::stablehlo::DynamicBroadcastInDimOp> {
  using OpConversionPattern::OpConversionPattern;

  // Try to recover the common dynamic broadcast shape pattern:
  //   %sa = shape.shape_of %A : tensor<...> -> tensor<rankxindex>
  //   %sb = shape.shape_of %B : tensor<...> -> tensor<rankxindex>
  //   %od = shape.broadcast %sa, %sb : ... -> tensor<rankxindex>
  //   %y  = stablehlo.dynamic_broadcast_in_dim %X, %od, dims=[0..rank-1]
  //
  // We lower this to:
  //   %od2 = mrt.broadcast_shape %A, %B : (tensor, tensor) -> !mrt.list<!mrt.i64>
  //   %y2  = mrt.expand %X, %od2
  //
  // This preserves the runtime-computed broadcasted shape and avoids encoding
  // dynamic extents as "sentinel constants".
  static Value buildRuntimeBroadcastSizeList(mlir::Value outputDimensions, mlir::Location loc,
                                             const mlir::TypeConverter *tc, ConversionPatternRewriter &rewriter) {
    if (!outputDimensions) return {};

    // Case 1: output_dimensions = shape.broadcast(shape_of(A), shape_of(B))
    if (auto bcast = outputDimensions.getDefiningOp<mlir::shape::BroadcastOp>()) {
      auto s0 = bcast->getOperand(0).getDefiningOp<mlir::shape::ShapeOfOp>();
      auto s1 = bcast->getOperand(1).getDefiningOp<mlir::shape::ShapeOfOp>();
      if (!s0 || !s1) return {};
      Value a = s0->getOperand(0);
      Value b = s1->getOperand(0);

      // Convert tensors to MRT tensor types if needed.
      auto convertTensor = [&](Value v) -> Value {
        Type dstType = tc ? tc->convertType(v.getType()) : Type();
        if (!dstType) return {};
        if (dstType == v.getType()) return v;
        // materializeTargetConversion is provided by the converter (uses unrealized_conversion_cast).
        return tc->materializeTargetConversion(rewriter, loc, dstType, v);
      };
      Value aMrt = convertTensor(a);
      Value bMrt = convertTensor(b);
      if (!aMrt || !bMrt) return {};

      // Result type: !mrt.list<!mrt.i64>
      auto listTy = mrt::ListType::get(rewriter.getContext(), mrt::I64Type::get(rewriter.getContext()));
      return rewriter.create<mrt::BroadcastShapeOp>(loc, listTy, aMrt, bMrt).getResult();
    }

    // Case 2: output_dimensions = shape.shape_of(A)  (rare but handle it)
    if (auto so = outputDimensions.getDefiningOp<mlir::shape::ShapeOfOp>()) {
      Value a = so->getOperand(0);
      Type dstType = tc ? tc->convertType(a.getType()) : Type();
      if (!dstType) return {};
      Value aMrt = (dstType == a.getType()) ? a : tc->materializeTargetConversion(rewriter, loc, dstType, a);
      if (!aMrt) return {};
      auto listTy = mrt::ListType::get(rewriter.getContext(), mrt::I64Type::get(rewriter.getContext()));
      return rewriter.create<mrt::ShapeOp>(loc, listTy, aMrt).getResult();
    }

    return {};
  }

  LogicalResult matchAndRewrite(mlir::stablehlo::DynamicBroadcastInDimOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type resultType = getTypeConverter()->convertType(op.getType());
    auto shapedType = dyn_cast<ShapedType>(resultType);
    if (!shapedType || !shapedType.hasRank()) {
      return failure();
    }

    auto inputTy = dyn_cast<RankedTensorType>(op.getOperand().getType());
    auto outputTy = dyn_cast<RankedTensorType>(op.getType());
    if (!inputTy || !outputTy) return failure();

    // Only handle same-rank broadcasts.
    if (inputTy.getRank() != outputTy.getRank()) return failure();

    // Require identity broadcast_dimensions [0..rank-1].
    auto dims = op.getBroadcastDimensions();
    if (static_cast<int64_t>(dims.size()) != inputTy.getRank()) return failure();
    for (int64_t i = 0; i < inputTy.getRank(); ++i) {
      if (dims[i] != i) return failure();
    }

    // Prefer preserving runtime shape computation from the output_dimensions chain.
    // This avoids encoding dynamic extents as sentinel constants and keeps the
    // broadcast semantics faithful in dynamic-shape graphs.
    Value sizeValue =
      buildRuntimeBroadcastSizeList(op.getOutputDimensions(), op.getLoc(), getTypeConverter(), rewriter);
    if (!sizeValue) {
      // Fallback: use the (converted) result type's shape. This may lose some
      // runtime broadcast information but keeps the pipeline robust.
      sizeValue = rewriter.create<mrt::CreateI64ArrayOp>(op.getLoc(), shapedType.getShape());
    }
    rewriter.replaceOpWithNewOp<mrt::ExpandOp>(op, resultType, adaptor.getOperand(), sizeValue);
    return success();
  }
};

// Convert stablehlo.transpose to mrt.permute
struct ConvertTransposeOp : public OpConversionPattern<mlir::stablehlo::TransposeOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::TransposeOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type resultType = getTypeConverter()->convertType(op.getResult().getType());

    auto inputType = dyn_cast<RankedTensorType>(op.getOperand().getType());
    if (!inputType || !inputType.hasRank()) {
      return failure();
    }

    // StableHLO permutation is an i64 array attribute (printed as array<i64: ...>).
    SmallVector<int64_t> perm;
    perm.reserve(inputType.getRank());
    auto permAttr = op.getPermutation();
    for (int64_t v : permAttr) {
      perm.push_back(v);
    }

    auto permValue = rewriter.create<mrt::CreateI64ArrayOp>(op.getLoc(), perm);
    rewriter.replaceOpWithNewOp<mrt::PermuteOp>(op, resultType, adaptor.getOperand(), permValue);
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
    registry.insert<mlir::tensor::TensorDialect>();
    registry.insert<mlir::shape::ShapeDialect>();
    // Allow mixed modules: Torch ops and TorchConversion bridge ops may remain
    // at this stage and be handled later by convert-torch-to-mrt.
    registry.insert<mlir::torch::Torch::TorchDialect>();
    registry.insert<mlir::torch::TorchConversion::TorchConversionDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();

    mopt::StablehloToMrtTypeConverter typeConverter(ctx);

    ConversionTarget target(*ctx);
    // Strict contract: this pass is responsible for eliminating StableHLO ops.
    // We only allow a *known* set of dialects to remain in the module. Any
    // unexpected dialect/op will cause conversion failure (surfacing missing
    // lowering early instead of silently passing through).
    target.addLegalDialect<mrt::MrtDialect>();
    target.addLegalDialect<mlir::arith::ArithDialect>();
    target.addLegalDialect<mlir::func::FuncDialect>();
    target.addLegalDialect<mlir::tensor::TensorDialect>();
    target.addLegalDialect<mlir::shape::ShapeDialect>();
    target.addLegalDialect<mlir::torch::Torch::TorchDialect>();
    target.addLegalDialect<mlir::torch::TorchConversion::TorchConversionDialect>();
    target.addLegalOp<mlir::ModuleOp>();
    // Allow unrealized conversion casts to pass through; they will be
    // reconciled later by reconcile-unrealized-casts pass.
    target.addLegalOp<mlir::UnrealizedConversionCastOp>();

    // Mark func.func as dynamically legal (need type conversion)
    target.addDynamicallyLegalOp<mlir::func::FuncOp>(
      [&](mlir::func::FuncOp op) { return typeConverter.isSignatureLegal(op.getFunctionType()); });
    target.addDynamicallyLegalOp<mlir::func::ReturnOp>(
      [&](mlir::func::ReturnOp op) { return typeConverter.isLegal(op.getOperandTypes()); });

    target.addIllegalDialect<mlir::stablehlo::StablehloDialect>();

    RewritePatternSet patterns(ctx);

    // Add simple binary op conversions using MOPT_CONVERT
    patterns.add<ConvertAddOp>(typeConverter, ctx);
    patterns.add<MOPT_CONVERT(mlir::stablehlo::MulOp, mrt::MulOp, Lhs, Rhs)>(typeConverter, ctx);
    patterns.add<MOPT_CONVERT(mlir::stablehlo::DivOp, mrt::DivOp, Lhs, Rhs)>(typeConverter, ctx);
    patterns.add<ConvertSubtractOp>(typeConverter, ctx);
    patterns.add<MOPT_CONVERT(mlir::stablehlo::RemOp, mrt::RemainderTensorTensorOp, Lhs, Rhs)>(typeConverter, ctx);
    patterns.add<MOPT_CONVERT(mlir::stablehlo::LogisticOp, mrt::SigmoidOp, Operand)>(typeConverter, ctx);

    // Add complex patterns
    patterns.add<ConvertReshapeOp>(typeConverter, ctx);
    patterns.add<ConvertConvertOp>(typeConverter, ctx);
    patterns.add<ConvertTransposeOp>(typeConverter, ctx);
    patterns.add<ConvertDotOp>(typeConverter, ctx);
    patterns.add<ConvertDynamicBroadcastInDimOp>(typeConverter, ctx);

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
