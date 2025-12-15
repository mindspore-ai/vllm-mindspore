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

#ifndef MOPT_FUSION_FUSION_STRATEGY_H
#define MOPT_FUSION_FUSION_STRATEGY_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mopt {

//===----------------------------------------------------------------------===//
// FusionStrategy - Abstract base class for fusion policies
//===----------------------------------------------------------------------===//

/// Abstract base class that defines the fusion policy for operations.
/// Different dialects can implement their own strategies.
class FusionStrategy {
 public:
  virtual ~FusionStrategy() = default;

  /// Returns whether this operation can be a fusion root.
  /// Fusion roots are the starting points for building fusion clusters.
  virtual bool isFusionRoot(mlir::Operation *op) const = 0;

  /// Returns whether this operation can be fused into a cluster.
  virtual bool canBeFused(mlir::Operation *op) const = 0;

  /// Returns whether this operation should never be fused.
  /// Such operations go through standard lowering path.
  virtual bool shouldNotBeFused(mlir::Operation *op) const = 0;
};

//===----------------------------------------------------------------------===//
// StablehloFusionStrategy - Fusion strategy for StableHLO dialect
//===----------------------------------------------------------------------===//

/// Fusion strategy implementation for StableHLO operations.
///
/// Fusion roots: Only elementwise operations (add, mul, tanh, etc.)
/// Note: Compute-intensive ops (dot_general, convolution, reduce) are NOT
/// fusion roots - they are handled by the standard MRT lowering path.
///
/// Fusible operations:
/// - Elementwise operations
/// - Broadcast operations
/// - Constants
///
/// Non-fusible operations:
/// - Compute-intensive ops (dot_general, convolution, reduce)
/// - Memory/communication ops (scatter, gather, collective ops)
class StablehloFusionStrategy : public FusionStrategy {
 public:
  bool isFusionRoot(mlir::Operation *op) const override {
    // Only elementwise ops are fusion roots
    return isElementwiseOp(op);
  }

  bool canBeFused(mlir::Operation *op) const override {
    // Elementwise operations
    if (isElementwiseOp(op)) {
      return true;
    }
    // Broadcast operations
    if (mlir::isa<mlir::stablehlo::BroadcastInDimOp>(op)) {
      return true;
    }
    // Constants can be fused
    if (mlir::isa<mlir::stablehlo::ConstantOp>(op)) {
      return true;
    }
    return false;
  }

  bool shouldNotBeFused(mlir::Operation *op) const override {
    // Dynamic shape ops should NOT be fused
    if (hasDynamicShape(op)) {
      return true;
    }
    // Compute-intensive ops should NOT be fused - they go through standard MRT lowering
    if (mlir::isa<mlir::stablehlo::DotGeneralOp, mlir::stablehlo::ConvolutionOp, mlir::stablehlo::ReduceOp>(op)) {
      return true;
    }
    // Memory/communication ops should NOT be fused
    return mlir::isa<mlir::stablehlo::ScatterOp, mlir::stablehlo::GatherOp, mlir::stablehlo::SortOp,
                     // Collective communication operations
                     mlir::stablehlo::AllGatherOp, mlir::stablehlo::AllReduceOp, mlir::stablehlo::AllToAllOp,
                     mlir::stablehlo::CollectiveBroadcastOp, mlir::stablehlo::CollectivePermuteOp,
                     mlir::stablehlo::ReduceScatterOp>(op);
  }

 private:
  /// Check if an operation is an elementwise operation
  static bool isElementwiseOp(mlir::Operation *op) {
    return mlir::isa<mlir::stablehlo::AddOp, mlir::stablehlo::SubtractOp, mlir::stablehlo::MulOp,
                     mlir::stablehlo::DivOp, mlir::stablehlo::MaxOp, mlir::stablehlo::MinOp, mlir::stablehlo::TanhOp,
                     mlir::stablehlo::ExpOp, mlir::stablehlo::LogOp, mlir::stablehlo::NegOp, mlir::stablehlo::AbsOp,
                     mlir::stablehlo::SqrtOp, mlir::stablehlo::RsqrtOp, mlir::stablehlo::CeilOp,
                     mlir::stablehlo::FloorOp, mlir::stablehlo::ClampOp, mlir::stablehlo::SelectOp,
                     mlir::stablehlo::CompareOp, mlir::stablehlo::ConvertOp>(op);
  }

  /// Check if an operation has dynamic shape in its inputs or outputs
  static bool hasDynamicShape(mlir::Operation *op) {
    // Check result types
    for (mlir::Type type : op->getResultTypes()) {
      if (auto shapedType = mlir::dyn_cast<mlir::ShapedType>(type)) {
        if (!shapedType.hasStaticShape()) {
          return true;
        }
      }
    }
    // Check operand types
    for (mlir::Value operand : op->getOperands()) {
      if (auto shapedType = mlir::dyn_cast<mlir::ShapedType>(operand.getType())) {
        if (!shapedType.hasStaticShape()) {
          return true;
        }
      }
    }
    return false;
  }
};

}  // namespace mopt

#endif  // MOPT_FUSION_FUSION_STRATEGY_H
