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

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "mopt/Dialect/Mrt/Mrt.h"
#include "mopt/Dialect/Mrt/MrtDialect.h"

using mlir::applyPatternsAndFoldGreedily;
using mlir::ArrayRef;
using mlir::cast;
using mlir::dyn_cast;
using mlir::failed;
using mlir::failure;
using mlir::LogicalResult;
using mlir::MLIRContext;
using mlir::ModuleOp;
using mlir::Operation;
using mlir::OperationPass;
using mlir::Pass;
using mlir::PassWrapper;
using mlir::Pattern;
using mlir::PatternRewriter;
using mlir::RankedTensorType;
using mlir::RewritePattern;
using mlir::RewritePatternSet;
using mlir::ShapedType;
using mlir::SmallVector;
using mlir::StringRef;
using mlir::success;
using mlir::Type;
using mlir::Value;
using mlir::arith::ConstantOp;

namespace {

// Import auto-generated pattern rewrite rules from TableGen
#include "StablehloToMrtPatterns.cpp.inc"

//===----------------------------------------------------------------------===//
// Helper functions
//===----------------------------------------------------------------------===//

// Create a constant tensor from shape values
Value createShapeTensor(PatternRewriter &rewriter, mlir::Location loc, ArrayRef<int64_t> shape) {
  auto shapeType = RankedTensorType::get({static_cast<int64_t>(shape.size())}, rewriter.getI64Type());
  auto shapeAttr = rewriter.getI64TensorAttr(shape);
  return rewriter.create<ConstantOp>(loc, shapeType, shapeAttr);
}

//===----------------------------------------------------------------------===//
// Conversion patterns
//===----------------------------------------------------------------------===//

// Convert stablehlo.reshape to mrt.reshape
struct ConvertReshapeOp : public RewritePattern {
  explicit ConvertReshapeOp(MLIRContext *context)
      : RewritePattern(mlir::stablehlo::ReshapeOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
    auto reshapeOp = cast<mlir::stablehlo::ReshapeOp>(op);

    auto resultType = dyn_cast<ShapedType>(reshapeOp.getResult().getType());
    if (!resultType || !resultType.hasRank()) {
      return failure();
    }

    // Create shape tensor
    SmallVector<int64_t> shape(resultType.getShape().begin(), resultType.getShape().end());
    auto shapeTensor = createShapeTensor(rewriter, op->getLoc(), shape);

    // Create mrt.reshape
    auto newOp = rewriter.create<mrt::ReshapeOp>(op->getLoc(), resultType, reshapeOp.getOperand(), shapeTensor);

    rewriter.replaceOp(op, newOp.getResult());
    return success();
  }
};

// Note: Simple one-to-one patterns (mul, logistic, dot, concatenate) are now
// defined in StablehloToMrtPatterns.td using DRR (Declarative Rewrite Rules)

// Convert stablehlo.batch_norm_inference to mrt.batch_norm
struct ConvertBatchNormInferenceOp : public RewritePattern {
  explicit ConvertBatchNormInferenceOp(MLIRContext *context)
      : RewritePattern(mlir::stablehlo::BatchNormInferenceOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
    auto bnOp = cast<mlir::stablehlo::BatchNormInferenceOp>(op);

    // Create mrt.batch_norm
    auto newOp =
      rewriter.create<mrt::BatchNormOp>(op->getLoc(), bnOp.getResult().getType(),
                                        bnOp.getOperand(),                                             // input
                                        bnOp.getScale(),                                               // scale
                                        bnOp.getOffset(),                                              // offset
                                        bnOp.getMean(),                                                // mean
                                        bnOp.getVariance(),                                            // variance
                                        rewriter.getF32FloatAttr(bnOp.getEpsilon().convertToFloat()),  // epsilon
                                        rewriter.getBoolAttr(false));  // is_training = false for inference

    rewriter.replaceOp(op, newOp.getResult());
    return success();
  }
};

// Convert stablehlo.maximum(x, 0) to mrt.relu
// This pattern detects the common ReLU pattern in StableHLO
struct ConvertMaximumToReluOp : public RewritePattern {
  explicit ConvertMaximumToReluOp(MLIRContext *context)
      : RewritePattern(mlir::stablehlo::MaxOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
    auto maxOp = cast<mlir::stablehlo::MaxOp>(op);

    // Check if one operand is a constant zero
    Value input;
    bool isRelu = false;

    // Check if rhs is constant zero
    if (auto constOp = maxOp.getRhs().getDefiningOp<mlir::stablehlo::ConstantOp>()) {
      if (auto denseAttr = dyn_cast<mlir::DenseElementsAttr>(constOp.getValue())) {
        if (denseAttr.isSplat()) {
          auto splatValue = denseAttr.getSplatValue<mlir::APFloat>();
          if (splatValue.isZero()) {
            input = maxOp.getLhs();
            isRelu = true;
          }
        }
      }
    }

    // Check if lhs is constant zero
    if (!isRelu) {
      if (auto constOp = maxOp.getLhs().getDefiningOp<mlir::stablehlo::ConstantOp>()) {
        if (auto denseAttr = dyn_cast<mlir::DenseElementsAttr>(constOp.getValue())) {
          if (denseAttr.isSplat()) {
            auto splatValue = denseAttr.getSplatValue<mlir::APFloat>();
            if (splatValue.isZero()) {
              input = maxOp.getRhs();
              isRelu = true;
            }
          }
        }
      }
    }

    if (!isRelu) {
      return failure();
    }

    // Create mrt.relu
    auto newOp = rewriter.create<mrt::ReluOp>(op->getLoc(), maxOp.getResult().getType(), input);

    rewriter.replaceOp(op, newOp.getResult());
    return success();
  }
};

// Convert stablehlo.convolution to mrt.conv
struct ConvertConvolutionOp : public RewritePattern {
  explicit ConvertConvolutionOp(MLIRContext *context)
      : RewritePattern(mlir::stablehlo::ConvolutionOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
    auto convOp = cast<mlir::stablehlo::ConvolutionOp>(op);

    // Extract attributes
    auto windowStrides = convOp.getWindowStridesAttr();
    auto padding = convOp.getPaddingAttr();
    // WARNING: lhsDilation is not used in the conversion
    auto rhsDilation = convOp.getRhsDilationAttr();

    // Convert to MRT format
    SmallVector<int64_t> strides;
    if (windowStrides) {
      for (auto stride : windowStrides.asArrayRef()) {
        strides.push_back(stride);
      }
    } else {
      // Default stride of 1
      auto resultType = dyn_cast<ShapedType>(convOp.getResult().getType());
      if (resultType) {
        int spatialDims = resultType.getRank() - 2;  // Assuming NCHW or similar
        strides.assign(spatialDims, 1);
      }
    }

    SmallVector<int64_t> paddingVec;
    if (padding) {
      for (auto val : padding.getValues<int64_t>()) {
        paddingVec.push_back(val);
      }
    }

    SmallVector<int64_t> dilation;
    if (rhsDilation) {
      for (auto val : rhsDilation.asArrayRef()) {
        dilation.push_back(val);
      }
    } else {
      int spatialDims = strides.size();
      dilation.assign(spatialDims, 1);
    }

    // Create mrt.conv (without bias for now)
    auto newOp = rewriter.create<mrt::ConvOp>(op->getLoc(), convOp.getResult().getType(),
                                              convOp.getLhs(),                       // input
                                              convOp.getRhs(),                       // kernel
                                              Value(),                               // bias (empty)
                                              rewriter.getI64ArrayAttr(strides),     // strides
                                              rewriter.getI64ArrayAttr(paddingVec),  // padding
                                              rewriter.getI64ArrayAttr(dilation),    // dilation
                                              rewriter.getBoolAttr(false));          // has_bias

    rewriter.replaceOp(op, newOp.getResult());
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
  }

  void runOnOperation() override {
    MLIRContext &context = getContext();
    RewritePatternSet patterns(&context);

    // Add DRR-generated patterns from StablehloToMrtPatterns.td
    populateWithGenerated(patterns);

    // Add complex C++ patterns that require custom logic
    patterns.add<ConvertReshapeOp>(&context);
    patterns.add<ConvertBatchNormInferenceOp>(&context);
    patterns.add<ConvertMaximumToReluOp>(&context);
    patterns.add<ConvertConvolutionOp>(&context);

    // Apply the patterns
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

namespace mlir {

std::unique_ptr<Pass> createConvertStablehloToMRTPass() { return std::make_unique<ConvertStablehloToMRTPass>(); }

}  // namespace mlir
