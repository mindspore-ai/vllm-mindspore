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

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mopt/Dialect/Mrt/Mrt.h"
#include "mopt/Dialect/Mrt/MrtDialect.h"

using mlir::applyPatternsAndFoldGreedily;
using mlir::dyn_cast;
using mlir::failed;
using mlir::failure;
using mlir::LogicalResult;
using mlir::MatchAnyOpTypeTag;
using mlir::MLIRContext;
using mlir::ModuleOp;
using mlir::Operation;
using mlir::OperationPass;
using mlir::Pass;
using mlir::PassWrapper;
using mlir::PatternRewriter;
using mlir::RankedTensorType;
using mlir::RewritePattern;
using mlir::RewritePatternSet;
using mlir::ShapedType;
using mlir::SmallVector;
using mlir::StringRef;
using mlir::success;
using mlir::Value;
using mlir::arith::ConstantOp;

namespace {

// Generic pattern to convert any reshape operation to mrt.reshape
// This pattern matches operations with name "reshape" that have:
// - One input tensor operand
// - One output tensor result
// - The output shape can be determined from the result type
struct ReshapeOpConversion : public RewritePattern {
  explicit ReshapeOpConversion(MLIRContext *context) : RewritePattern(MatchAnyOpTypeTag(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
    // Check if this is a reshape operation
    if (op->getName().getStringRef() != "stablehlo.reshape" && op->getName().getStringRef() != "reshape") {
      return failure();
    }

    // Check if it has exactly one input and one output
    if (op->getNumOperands() != 1 || op->getNumResults() != 1) {
      return failure();
    }

    // Get the input tensor
    Value input = op->getOperand(0);

    // Get the result type
    auto resultType = mlir::dyn_cast<ShapedType>(op->getResult(0).getType());
    if (!resultType) {
      return failure();
    }

    // Extract the shape from the result type
    SmallVector<int64_t> shape(resultType.getShape().begin(), resultType.getShape().end());

    // Create a shape tensor (1D tensor with shape values)
    auto shapeType = RankedTensorType::get({static_cast<int64_t>(shape.size())}, rewriter.getI64Type());
    auto shapeAttr = rewriter.getI64TensorAttr(shape);
    auto shapeTensor = rewriter.create<ConstantOp>(op->getLoc(), shapeType, shapeAttr);

    // Create the new mrt.reshape operation
    auto newOp = rewriter.create<mrt::ReshapeOp>(op->getLoc(), resultType, input, shapeTensor);

    // Replace the old operation with the new one
    rewriter.replaceOp(op, newOp.getResult());

    return success();
  }
};

// Pass to convert reshape operations to mrt.reshape
struct ConvertReshapeToMRTPass : public PassWrapper<ConvertReshapeToMRTPass, OperationPass<ModuleOp>> {
  StringRef getArgument() const final { return "convert-reshape-to-mrt"; }
  StringRef getDescription() const final { return "Convert reshape operations to MRT reshape operations"; }

  void runOnOperation() override {
    MLIRContext &context = getContext();
    RewritePatternSet patterns(&context);

    // Add the reshape conversion pattern
    patterns.add<ReshapeOpConversion>(&context);

    // Apply the patterns
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) signalPassFailure();
  }
};

}  // namespace

namespace mlir {

std::unique_ptr<Pass> createConvertReshapeToMRTPass() { return std::make_unique<ConvertReshapeToMRTPass>(); }

}  // namespace mlir
