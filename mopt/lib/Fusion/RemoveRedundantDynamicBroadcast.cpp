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

#include "mopt/Fusion/Passes.h"
#include "mopt/Fusion/SymbolicShape.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "remove-redundant-dynamic-broadcast"

namespace mlir {
namespace {
using mlir::mopt::fusion::getSymbolicShapeSignature;
using mlir::mopt::fusion::kSymbolicShapeAttrName;

/// Check if a dynamic_broadcast_in_dim is a no-op identity broadcast.
/// This is true when:
/// 1. The broadcast_dimensions are identity [0, 1, ..., rank-1]
/// 2. The input and output have the same rank
/// 3. Replacing the broadcast result with its input is type-safe (same tensor type)
///
/// Note: In dynamic shape scenarios, even an "identity dims" broadcast may still
/// semantically change the runtime shape. We only use this predicate as a *necessary*
/// condition for elimination, and pair it with stronger proofs (e.g. symbolic shape
/// equality / self-shape-of) in individual rewrite patterns.
static bool isIdentityBroadcast(stablehlo::DynamicBroadcastInDimOp op) {
  auto inputType = dyn_cast<RankedTensorType>(op.getOperand().getType());
  auto outputType = dyn_cast<RankedTensorType>(op.getType());
  if (!inputType || !outputType) return false;

  if (inputType.getRank() != outputType.getRank()) return false;

  // If types differ, replacing broadcast with input will immediately produce
  // illegal IR (users expect the broadcast result type).
  if (inputType != outputType) return false;

  auto dims = op.getBroadcastDimensions();
  if (static_cast<int64_t>(dims.size()) != inputType.getRank()) return false;
  for (int64_t i = 0; i < inputType.getRank(); ++i) {
    if (dims[i] != i) return false;
  }
  return true;
}

/// Collect all dynamic_broadcast_in_dim ops in a shape.assuming region
/// and check if they can all be eliminated.
///
/// Returns true if all broadcasts in the region are identity broadcasts
/// AND all their source tensors have matching symbolic shapes.
static bool canEliminateBroadcastsInAssuming(shape::AssumingOp assumingOp,
                                             SmallVectorImpl<stablehlo::DynamicBroadcastInDimOp> &broadcasts,
                                             DictionaryAttr &commonSig) {
  broadcasts.clear();
  commonSig = nullptr;

  Region &region = assumingOp.getDoRegion();
  if (region.empty()) return false;

  Block &block = region.front();

  for (Operation &op : block) {
    if (auto broadcastOp = dyn_cast<stablehlo::DynamicBroadcastInDimOp>(&op)) {
      if (!isIdentityBroadcast(broadcastOp)) {
        LLVM_DEBUG(llvm::dbgs() << "[RemoveDynBcast] Non-identity broadcast found, skipping\n");
        return false;
      }

      Value input = broadcastOp.getOperand();
      DictionaryAttr sig = getSymbolicShapeSignature(input);

      if (!sig) {
        LLVM_DEBUG(llvm::dbgs() << "[RemoveDynBcast] No symbolic shape on broadcast input\n");
        return false;
      }

      if (!commonSig) {
        commonSig = sig;
      } else if (commonSig != sig) {
        LLVM_DEBUG(llvm::dbgs() << "[RemoveDynBcast] Mismatched symbolic shapes\n");
        return false;
      }

      broadcasts.push_back(broadcastOp);
    }
  }

  if (broadcasts.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "[RemoveDynBcast] No dynamic broadcasts in assuming region\n");
    return false;
  }

  LLVM_DEBUG(llvm::dbgs() << "[RemoveDynBcast] Found " << broadcasts.size() << " eliminable broadcasts\n");
  return true;
}

/// Pattern to remove redundant shape.assuming regions containing
/// dynamic_broadcast_in_dim operations that are provably no-ops.
struct RemoveRedundantAssumingPattern : public OpRewritePattern<shape::AssumingOp> {
  using OpRewritePattern<shape::AssumingOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(shape::AssumingOp assumingOp, PatternRewriter &rewriter) const override {
    SmallVector<stablehlo::DynamicBroadcastInDimOp> broadcasts;
    DictionaryAttr commonSig;

    if (!canEliminateBroadcastsInAssuming(assumingOp, broadcasts, commonSig)) {
      return failure();
    }

    Region &region = assumingOp.getDoRegion();
    Block &block = region.front();

    // Build mapping from broadcast results to their original inputs
    IRMapping mapping;
    for (auto broadcastOp : broadcasts) {
      mapping.map(broadcastOp.getResult(), broadcastOp.getOperand());
    }

    // Find the yield op to get the results
    auto yieldOp = cast<shape::AssumingYieldOp>(block.getTerminator());

    // Clone non-broadcast operations before the assuming op, replacing
    // broadcast results with their inputs
    rewriter.setInsertionPoint(assumingOp);
    SmallVector<Value> newResults;

    for (Operation &op : block) {
      if (isa<stablehlo::DynamicBroadcastInDimOp>(&op)) {
        continue;
      }
      if (isa<shape::AssumingYieldOp>(&op)) {
        // Map yield operands to final results
        for (Value yieldOperand : yieldOp.getOperands()) {
          Value mapped = mapping.lookupOrDefault(yieldOperand);
          newResults.push_back(mapped);
        }
        continue;
      }
      if (isa<shape::BroadcastOp>(&op)) {
        continue;
      }

      // Clone the operation with mapped operands
      Operation *clonedOp = rewriter.clone(op, mapping);

      // Update mapping for results
      for (auto [oldResult, newResult] : llvm::zip(op.getResults(), clonedOp->getResults())) {
        mapping.map(oldResult, newResult);
      }

      // Propagate symbolic shape attribute if the original op had it
      if (DictionaryAttr sig = op.getAttrOfType<DictionaryAttr>(kSymbolicShapeAttrName)) {
        clonedOp->setAttr(kSymbolicShapeAttrName, sig);
      } else if (commonSig) {
        clonedOp->setAttr(kSymbolicShapeAttrName, commonSig);
      }
    }

    // Replace uses of assuming results with new results
    for (auto [oldResult, newResult] : llvm::zip(assumingOp.getResults(), newResults)) {
      rewriter.replaceAllUsesWith(oldResult, newResult);
    }

    // Erase the assuming op
    rewriter.eraseOp(assumingOp);

    LLVM_DEBUG(llvm::dbgs() << "[RemoveDynBcast] Successfully removed assuming region\n");
    return success();
  }
};

/// Pattern to eliminate identity dynamic_broadcast_in_dim when:
/// 1. The broadcast is identity (dims = [0, 1, ..., rank-1])
/// 2. The input already has matching symbolic shape, OR
/// 3. Multiple broadcasts share the same output_dimensions from shape.broadcast
///
/// This handles the case where dynamic_broadcast_in_dim appears outside of
/// shape.assuming regions (e.g., after fusion outlining).
struct RemoveIdentityDynamicBroadcast : public OpRewritePattern<stablehlo::DynamicBroadcastInDimOp> {
  using OpRewritePattern<stablehlo::DynamicBroadcastInDimOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::DynamicBroadcastInDimOp op, PatternRewriter &rewriter) const override {
    if (!isIdentityBroadcast(op)) {
      return failure();
    }

    Value input = op.getOperand();
    Value outputDims = op.getOutputDimensions();

    // Case 1: output_dimensions comes from shape.shape_of on the same input
    // This means the broadcast is definitely a no-op
    if (auto shapeOf = outputDims.getDefiningOp<shape::ShapeOfOp>()) {
      if (shapeOf.getArg() == input) {
        LLVM_DEBUG(llvm::dbgs() << "[RemoveDynBcast] Removing self-broadcast\n");
        rewriter.replaceOp(op, input);
        return success();
      }
    }

    // Case 2: output_dimensions comes from shape.broadcast of two shape_of ops
    // If both inputs to shape.broadcast are shape_of on tensors with same rank,
    // and this broadcast_in_dim is identity, the broadcast is a no-op
    if (auto shapeBroadcast = outputDims.getDefiningOp<shape::BroadcastOp>()) {
      auto shapes = shapeBroadcast.getShapes();
      if (shapes.size() == 2) {
        auto shapeOf0 = shapes[0].getDefiningOp<shape::ShapeOfOp>();
        auto shapeOf1 = shapes[1].getDefiningOp<shape::ShapeOfOp>();

        if (shapeOf0 && shapeOf1) {
          Value arg0 = shapeOf0.getArg();
          Value arg1 = shapeOf1.getArg();

          // If this broadcast's input is one of the shape_of args, it's a no-op
          if (input == arg0 || input == arg1) {
            auto type0 = dyn_cast<RankedTensorType>(arg0.getType());
            auto type1 = dyn_cast<RankedTensorType>(arg1.getType());

            if (type0 && type1 && type0.getRank() == type1.getRank()) {
              LLVM_DEBUG(llvm::dbgs() << "[RemoveDynBcast] Removing broadcast with matching ranks\n");
              rewriter.replaceOp(op, input);
              return success();
            }
          }
        }
      }
    }

    // Case 3: Check symbolic shape attribute
    DictionaryAttr inputSig = getSymbolicShapeSignature(input);
    if (inputSig) {
      // If input has symbolic shape and output_dimensions also derives from
      // a tensor with the same symbolic shape, we can eliminate
      if (auto shapeOf = outputDims.getDefiningOp<shape::ShapeOfOp>()) {
        DictionaryAttr dimsSig = getSymbolicShapeSignature(shapeOf.getArg());
        if (dimsSig && inputSig == dimsSig) {
          LLVM_DEBUG(llvm::dbgs() << "[RemoveDynBcast] Removing broadcast with matching symbolic shapes\n");
          rewriter.replaceOp(op, input);
          return success();
        }
      }
    }

    return failure();
  }
};

/// Pattern to remove orphaned shape.cstr_broadcastable ops after
/// their assuming regions are eliminated.
struct RemoveOrphanedCstrBroadcastable : public OpRewritePattern<shape::CstrBroadcastableOp> {
  using OpRewritePattern<shape::CstrBroadcastableOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(shape::CstrBroadcastableOp op, PatternRewriter &rewriter) const override {
    if (op.use_empty()) {
      rewriter.eraseOp(op);
      return success();
    }
    return failure();
  }
};

/// Pattern to remove orphaned shape.shape_of ops.
struct RemoveOrphanedShapeOf : public OpRewritePattern<shape::ShapeOfOp> {
  using OpRewritePattern<shape::ShapeOfOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(shape::ShapeOfOp op, PatternRewriter &rewriter) const override {
    if (op.use_empty()) {
      rewriter.eraseOp(op);
      return success();
    }
    return failure();
  }
};

/// Pattern to remove orphaned shape.broadcast ops.
struct RemoveOrphanedBroadcast : public OpRewritePattern<shape::BroadcastOp> {
  using OpRewritePattern<shape::BroadcastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(shape::BroadcastOp op, PatternRewriter &rewriter) const override {
    if (op.use_empty()) {
      rewriter.eraseOp(op);
      return success();
    }
    return failure();
  }
};

}  // namespace
}  // namespace mlir

namespace mlir {

#define GEN_PASS_DEF_REMOVEREDUNDANTDYNAMICBROADCAST
#include "mopt/Fusion/Passes.h.inc"

struct RemoveRedundantDynamicBroadcastPass
    : public impl::RemoveRedundantDynamicBroadcastBase<RemoveRedundantDynamicBroadcastPass> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    MLIRContext *ctx = func.getContext();

    RewritePatternSet patterns(ctx);
    patterns.add<RemoveRedundantAssumingPattern>(ctx);
    patterns.add<RemoveIdentityDynamicBroadcast>(ctx);
    patterns.add<RemoveOrphanedCstrBroadcastable>(ctx);
    patterns.add<RemoveOrphanedShapeOf>(ctx);
    patterns.add<RemoveOrphanedBroadcast>(ctx);

    if (failed(applyPatternsGreedily(func, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<Pass> createRemoveRedundantDynamicBroadcastPass() {
  return std::make_unique<RemoveRedundantDynamicBroadcastPass>();
}

}  // namespace mlir
