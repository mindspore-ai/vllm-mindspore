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

#include <algorithm>
#include <iterator>
#include <numeric>
#include <optional>

#include "mopt/Fusion/FusionStrategy.h"
#include "mopt/Fusion/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionOps.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "stablehlo-fusion"

using mlir::ArrayRef;
using mlir::dyn_cast;
using mlir::func::FuncOp;
using mlir::IRMapping;
using mlir::isa;
using mlir::ModuleOp;
using mlir::Operation;
using mlir::OperationPass;
using mlir::OpBuilder;
using mlir::Pass;
using mlir::PassWrapper;
using mlir::SmallVector;
using mlir::StringRef;
using mlir::Type;
using mlir::Value;
using llvm::SetVector;

namespace {

//===----------------------------------------------------------------------===//
// High-level algorithm overview (design notes)
//===----------------------------------------------------------------------===//
//
// This pass outlines locally fusible StableHLO compute chains into separate `func.func`
// functions, and replaces the original ops with a `func.call`.
//
// Overview:
// - Step 0 (pre-clean): remove metadata-only Torch symbolic-shape bridge chains that
//   would otherwise inflate cluster live-outs.
// - Step 1 (cluster build): for each fusion-root (policy-defined), grow a cluster by:
//   - fusing producers (operands) when it is "single-consumer within cluster" (with
//     shape-auxiliary users ignored), and
//   - fusing consumers (users) only when they do not depend on defining ops outside
//     the current cluster (conservative, locally-connected growth).
// - Step 2 (finalize IO): inputs are operands defined outside the cluster; outputs are
//   results used outside the cluster.
// - Step 3 (outline): clone cluster ops into a new function in topological order, return
//   the cluster outputs, and replace original uses with call results.
//
// NOTE: This file intentionally keeps the fusion-growth rules conservative. This is a
// readability-oriented pass; broader fusion requires extra dominance/legality checks.

//===----------------------------------------------------------------------===//
// FusionCluster - Data structure for a cluster of fusible operations
//===----------------------------------------------------------------------===//

/// A cluster of operations that can be fused together
struct FusionCluster {
  SetVector<Operation *> ops;      // Operations in the cluster
  SmallVector<Value> inputs;       // External inputs to the cluster
  SmallVector<Value> outputs;      // Outputs from the cluster
  SmallVector<Type> outputTypes;   // Types of outputs
  std::string parentFuncName;      // Name of the parent function containing this cluster
};

//===----------------------------------------------------------------------===//
// Helper functions for fusion analysis
//===----------------------------------------------------------------------===//

/// Check if an operation is a shape-related auxiliary op that should be
/// ignored when checking "all users in cluster" constraint.
/// These ops don't participate in actual computation, they only extract
/// shape info or perform type conversion for shape propagation.
static bool isShapeAuxiliaryOp(Operation *op) {
  return isa<mlir::shape::ShapeOfOp, mlir::torch::TorchConversion::FromBuiltinTensorOp,
             mlir::torch::TorchConversion::ToBuiltinTensorOp, mlir::torch::Torch::BindSymbolicShapeOp>(op);
}

/// Returns true if `v` is a block argument (i.e., not produced by an op).
static bool isBlockArgument(Value v) { return v && v.getDefiningOp() == nullptr; }

/// Can we pull `producer` into the cluster as a producer of `consumer`?
///
/// This helper preserves the original (pre-refactor) rule used by this pass:
/// - Fuse `producer` if ALL users of `producer->getResult(0)` are either:
///   - the current `consumer`, or
///   - already in the cluster, or
///   - shape-auxiliary users (ignored for the single-consumer constraint), OR
/// - `producer` is a stablehlo.constant.
static bool canFuseProducer(Operation *producer, Operation *consumer, const SetVector<Operation *> &clusterOps) {
  if (!producer || !consumer) return false;
  if (isa<mlir::stablehlo::ConstantOp>(producer)) return true;
  if (producer->getNumResults() == 0) return false;
  Value result0 = producer->getResult(0);
  for (Operation *user : result0.getUsers()) {
    if (user == consumer) continue;
    if (clusterOps.contains(user)) continue;
    if (isShapeAuxiliaryOp(user)) continue;
    return false;
  }
  return true;
}

/// Can we pull `user` into the cluster as a consumer of `edgeValue`?
///
/// Preserves the original conservative growth rule: we only fuse `user` when all
/// of its other operands are either block arguments or produced by ops already in
/// the cluster (i.e., we do not expand the cluster through "external defining op"
/// edges).
static bool canFuseConsumer(Operation *user, Value edgeValue, const SetVector<Operation *> &clusterOps) {
  if (!user) return false;
  for (Value opnd : user->getOperands()) {
    if (opnd == edgeValue) continue;
    if (isBlockArgument(opnd)) continue;
    Operation *def = opnd.getDefiningOp();
    if (def && clusterOps.contains(def)) continue;
    return false;
  }
  return true;
}

/// Safely compare if operation a is before operation b in the same block.
/// Returns false if they are in different blocks or if either is null.
static bool safeIsBeforeInBlock(Operation *a, Operation *b) {
  if (!a || !b) {
    return false;
  }
  if (a->getBlock() != b->getBlock()) {
    return false;
  }
  return a->isBeforeInBlock(b);
}

struct CallInsertionPlan {
  Operation *insertionPoint = nullptr;
  bool insertAfter = false;
};

static llvm::SmallPtrSet<Operation *, 16> buildClusterOpsSet(const FusionCluster &cluster) {
  llvm::SmallPtrSet<Operation *, 16> set;
  set.reserve(cluster.ops.size());
  for (Operation *op : cluster.ops) {
    set.insert(op);
  }
  return set;
}

static Operation *findLastInputDefOutsideCluster(const FusionCluster &cluster,
                                                 const llvm::SmallPtrSetImpl<Operation *> &clusterOpsSet) {
  Operation *lastInputDef = nullptr;
  for (Value input : cluster.inputs) {
    Operation *defOp = input.getDefiningOp();
    if (!defOp || clusterOpsSet.contains(defOp)) continue;
    if (!lastInputDef || safeIsBeforeInBlock(lastInputDef, defOp)) {
      lastInputDef = defOp;
    }
  }
  return lastInputDef;
}

static Operation *findFirstExternalUserOfOutputs(const FusionCluster &cluster,
                                                 const llvm::SmallPtrSetImpl<Operation *> &clusterOpsSet) {
  Operation *firstExternalUser = nullptr;
  for (Value output : cluster.outputs) {
    for (Operation *user : output.getUsers()) {
      if (clusterOpsSet.contains(user)) continue;
      if (!firstExternalUser || safeIsBeforeInBlock(user, firstExternalUser)) {
        firstExternalUser = user;
      }
    }
  }
  return firstExternalUser;
}

static Operation *findLastOpInCluster(const FusionCluster &cluster) {
  return std::accumulate(
    cluster.ops.begin(), cluster.ops.end(), static_cast<Operation *>(nullptr),
    [](Operation *best, Operation *op) { return (!best || safeIsBeforeInBlock(best, op)) ? op : best; });
}

static std::optional<CallInsertionPlan> computeCallInsertionPlan(
  const FusionCluster &cluster, const llvm::SmallPtrSetImpl<Operation *> &clusterOpsSet) {
  Operation *lastInputDef = findLastInputDefOutsideCluster(cluster, clusterOpsSet);
  Operation *firstExternalUser = findFirstExternalUserOfOutputs(cluster, clusterOpsSet);

  CallInsertionPlan plan;
  if (lastInputDef && firstExternalUser) {
    // Verify SSA property: lastInputDef must be before firstExternalUser.
    if (!safeIsBeforeInBlock(lastInputDef, firstExternalUser)) {
      LLVM_DEBUG(llvm::dbgs() << "[Fusion] SSA violation: input defined after user or in different blocks\n");
      return std::nullopt;
    }
    plan.insertionPoint = lastInputDef;
    plan.insertAfter = true;
    return plan;
  }

  if (lastInputDef) {
    plan.insertionPoint = lastInputDef;
    plan.insertAfter = true;
    return plan;
  }

  if (firstExternalUser) {
    plan.insertionPoint = firstExternalUser;
    plan.insertAfter = false;
    return plan;
  }

  plan.insertionPoint = findLastOpInCluster(cluster);
  plan.insertAfter = true;
  if (!plan.insertionPoint) {
    LLVM_DEBUG(llvm::dbgs() << "[Fusion] Could not find insertion point\n");
    return std::nullopt;
  }
  return plan;
}

/// Remove Torch symbolic-shape modeling ops that only serve as metadata and
/// would otherwise force outlined StableHLO clusters to return multiple values.
///
/// Pattern:
///   %t = torch_c.from_builtin_tensor %x
///   torch.bind_symbolic_shape %t, ...
/// If %t is ONLY used by torch.bind_symbolic_shape, the bind has no runtime
/// semantic effect and the bridge value %t is dead. We can erase both ops.
///
/// This is important for the DVM-call lowering path: even though runtime dvm_call
/// can return multiple outputs, keeping metadata-only bridge values alive can
/// unnecessarily increase outlined live-outs and hinder downstream lowering.
/// In practice, this also improves robustness because some downstream runtimes
/// still assume a single "primary" output shape when plumbing dyn-shape metadata.
static void cleanupDeadTorchSymbolicShapeBinds(FuncOp funcOp) {
  using mlir::torch::Torch::BindSymbolicShapeOp;
  using mlir::torch::TorchConversion::FromBuiltinTensorOp;

  SmallVector<Operation *> opsToErase;

  funcOp.walk([&](FromBuiltinTensorOp fromOp) {
    Value vt = fromOp.getResult();
    if (vt.use_empty()) {
      opsToErase.push_back(fromOp.getOperation());
      return;
    }

    // Check if ALL users are bind_symbolic_shape.
    for (Operation *user : vt.getUsers()) {
      if (!isa<BindSymbolicShapeOp>(user)) {
        return;
      }
    }

    // Erase all bind ops first, then erase the bridge op.
    auto users = llvm::make_early_inc_range(vt.getUsers());
    std::copy(users.begin(), users.end(), std::back_inserter(opsToErase));
    opsToErase.push_back(fromOp.getOperation());
  });

  if (opsToErase.empty()) return;

  // Deterministic erase order: erase users first where possible.
  // Also deduplicate in case multiple walks added the same op.
  llvm::SmallDenseSet<Operation *, 16> seen;
  SmallVector<Operation *> unique;
  unique.reserve(opsToErase.size());
  for (Operation *op : opsToErase) {
    if (op && !seen.contains(op)) {
      seen.insert(op);
      unique.push_back(op);
    }
  }
  std::sort(unique.begin(), unique.end(), [](Operation *a, Operation *b) { return safeIsBeforeInBlock(b, a); });

  for (Operation *op : unique) {
    if (op && op->use_empty()) {
      op->erase();
    }
  }
}

//===----------------------------------------------------------------------===//
// FusionClusterBuilder - Builds fusion clusters using a given strategy
//===----------------------------------------------------------------------===//

/// Builds fusion clusters from a module using the provided fusion strategy.
class FusionClusterBuilder {
 public:
  explicit FusionClusterBuilder(const mopt::FusionStrategy &strategy) : strategy_(strategy) {}

  /// Identify all fusion clusters in the module
  SmallVector<FusionCluster, 4> buildClusters(ModuleOp module) {
    SmallVector<FusionCluster, 4> clusters;
    SetVector<Operation *> allVisited;

    // Debug counters
    int totalOps = 0;
    int fusionRoots = 0;

    // Walk through each function separately to track parent function names
    module.walk([&](FuncOp funcOp) {
      std::string parentFuncName = funcOp.getName().str();

      funcOp.walk([&](Operation *op) {
        totalOps++;

        if (strategy_.isFusionRoot(op) && !allVisited.contains(op)) {
          fusionRoots++;
          FusionCluster cluster;
          SetVector<Operation *> clusterVisited;

          collectFusibleOps(op, cluster, clusterVisited);

          // Skip single-operation clusters (no fusion benefit)
          if (cluster.ops.size() <= 1) {
            return;
          }

          // Mark all ops as globally visited
          for (Operation *clusterOp : cluster.ops) {
            allVisited.insert(clusterOp);
          }

          // Identify inputs and outputs
          finalizeClusterIO(cluster);
          cluster.parentFuncName = parentFuncName;
          clusters.push_back(std::move(cluster));
        }
      });
    });

    LLVM_DEBUG(llvm::dbgs() << "[Fusion] Total ops examined: " << totalOps
                            << ", fusion roots found: " << fusionRoots
                            << ", clusters created: " << clusters.size() << "\n");

    return clusters;
  }

 private:
  const mopt::FusionStrategy &strategy_;

  /// Check if a value has users outside the given set of operations
  static bool hasExternalUsers(Value value, const SetVector<Operation *> &clusterOps) {
    for (Operation *user : value.getUsers()) {
      if (!clusterOps.contains(user)) {
        return true;
      }
    }
    return false;
  }

  /// Collect fusible operations starting from a root
  void collectFusibleOps(Operation *root, FusionCluster &cluster, SetVector<Operation *> &visited) {
    if (visited.contains(root)) {
      return;
    }
    visited.insert(root);
    cluster.ops.insert(root);

    fuseProducers(root, cluster, visited);
    fuseConsumers(root, cluster, visited);
  }

  void fuseProducers(Operation *root, FusionCluster &cluster, SetVector<Operation *> &visited) {
    for (Value operand : root->getOperands()) {
      Operation *defOp = operand.getDefiningOp();
      if (!defOp) {
        continue;  // Block argument
      }

      if (!strategy_.canBeFused(defOp) || strategy_.shouldNotBeFused(defOp)) {
        continue;
      }

      if (canFuseProducer(defOp, root, cluster.ops)) {
        collectFusibleOps(defOp, cluster, visited);
      }
    }
  }

  void fuseConsumers(Operation *root, FusionCluster &cluster, SetVector<Operation *> &visited) {
    for (Value result : root->getResults()) {
      for (Operation *user : result.getUsers()) {
        if (!strategy_.canBeFused(user) || strategy_.shouldNotBeFused(user) || visited.contains(user)) {
          continue;
        }

        if (canFuseConsumer(user, result, cluster.ops)) {
          collectFusibleOps(user, cluster, visited);
        }
      }
    }
  }

  /// Finalize cluster by identifying inputs and outputs
  void finalizeClusterIO(FusionCluster &cluster) {
    SetVector<Value> inputSet;
    SetVector<Value> outputSet;

    for (Operation *clusterOp : cluster.ops) {
      // Find external inputs
      for (Value operand : clusterOp->getOperands()) {
        Operation *defOp = operand.getDefiningOp();
        if (!defOp || !cluster.ops.contains(defOp)) {
          inputSet.insert(operand);
        }
      }

      // Find outputs (results used outside the cluster)
      for (Value result : clusterOp->getResults()) {
        if (hasExternalUsers(result, cluster.ops)) {
          outputSet.insert(result);
        }
      }
    }

    cluster.inputs = inputSet.takeVector();
    cluster.outputs = outputSet.takeVector();
    for (Value output : cluster.outputs) {
      cluster.outputTypes.push_back(output.getType());
    }
  }
};

//===----------------------------------------------------------------------===//
// FunctionOutliner - Outlines fusion clusters to separate functions
//===----------------------------------------------------------------------===//

/// Handles outlining of fusion clusters to separate functions and replacing
/// the original operations with function calls.
class FunctionOutliner {
 public:
  explicit FunctionOutliner(ModuleOp module) : module_(module) {}

  /// Outline a fusion cluster to a separate function
  FuncOp outline(FusionCluster &cluster, unsigned clusterIndex, StringRef parentFuncName) {
    if (cluster.ops.empty()) {
      return nullptr;
    }

    // Get the first operation's location for the function
    Operation *firstOp = *cluster.ops.begin();
    mlir::Location loc = firstOp->getLoc();

    // Build function type
    SmallVector<Type> inputTypes;
    std::transform(cluster.inputs.begin(), cluster.inputs.end(), std::back_inserter(inputTypes),
                   [](Value input) { return input.getType(); });

    auto funcType = mlir::FunctionType::get(module_.getContext(), inputTypes, cluster.outputTypes);

    // Create function name with parent function name as prefix for uniqueness and traceability
    std::string funcName = (parentFuncName + "_fused_cluster_" + std::to_string(clusterIndex)).str();

    // Create function at module level
    OpBuilder moduleBuilder(module_.getBodyRegion());
    auto funcOp = FuncOp::create(moduleBuilder, loc, funcName, funcType);
    funcOp.setPrivate();

    // Mark with fusion.outlined attribute
    funcOp->setAttr("fusion.outlined", mlir::UnitAttr::get(module_.getContext()));

    // Create entry block
    mlir::Block *entryBlock = funcOp.addEntryBlock();
    OpBuilder funcBuilder(entryBlock, entryBlock->begin());

    // Map old values to new values
    IRMapping mapper;

    // Map inputs to block arguments
    for (size_t i = 0; i < cluster.inputs.size(); ++i) {
      mapper.map(cluster.inputs[i], entryBlock->getArgument(i));
    }

    // Topologically sort operations for correct order
    SmallVector<Operation *> sortedOps = topologicalSort(cluster.ops);

    // Check if topological sort failed (cycle detected)
    if (sortedOps.empty() && !cluster.ops.empty()) {
      LLVM_DEBUG(llvm::dbgs() << "[Fusion] Skipping cluster due to cycle in dependency graph\n");
      funcOp.erase();
      return nullptr;
    }

    // Clone operations into the function
    for (Operation *op : sortedOps) {
      Operation *clonedOp = funcBuilder.clone(*op, mapper);
      for (size_t i = 0; i < op->getNumResults(); ++i) {
        mapper.map(op->getResult(i), clonedOp->getResult(i));
      }
    }

    // Create return
    SmallVector<Value> returnValues;
    std::transform(cluster.outputs.begin(), cluster.outputs.end(), std::back_inserter(returnValues),
                   [&mapper](Value output) { return mapper.lookup(output); });
    mlir::func::ReturnOp::create(funcBuilder, loc, returnValues);

    return funcOp;
  }

  /// Replace a fusion cluster with a call to the outlined function
  void replaceWithCall(FusionCluster &cluster, FuncOp outlinedFunc) {
    if (cluster.ops.empty() || !outlinedFunc) {
      return;
    }

    // Find insertion point with SSA dominance constraints.
    llvm::SmallPtrSet<Operation *, 16> clusterOpsSet = buildClusterOpsSet(cluster);
    auto plan = computeCallInsertionPlan(cluster, clusterOpsSet);
    if (!plan) return;

    // Create call at the insertion point
    OpBuilder builder(plan->insertionPoint);
    if (plan->insertAfter) {
      builder.setInsertionPointAfter(plan->insertionPoint);
    } else {
      builder.setInsertionPoint(plan->insertionPoint);
    }

    auto callOp = mlir::func::CallOp::create(builder, plan->insertionPoint->getLoc(), outlinedFunc, cluster.inputs);

    // Replace uses of cluster outputs with call results
    for (size_t i = 0; i < cluster.outputs.size(); ++i) {
      cluster.outputs[i].replaceAllUsesExcept(callOp.getResult(i), clusterOpsSet);
    }

    // Erase cluster operations in reverse order (last to first in block)
    SmallVector<Operation *> sortedOps(cluster.ops.begin(), cluster.ops.end());
    std::sort(sortedOps.begin(), sortedOps.end(), [](Operation *a, Operation *b) { return safeIsBeforeInBlock(b, a); });

    for (Operation *op : sortedOps) {
      op->erase();
    }
  }

 private:
  ModuleOp module_;

  /// Topologically sort operations for correct cloning order using Kahn's algorithm.
  /// Time complexity: O(V + E) where V is number of ops and E is number of dependencies.
  /// Returns empty vector if a cycle is detected.
  static SmallVector<Operation *> topologicalSort(const SetVector<Operation *> &ops) {
    SmallVector<Operation *> sortedOps;
    sortedOps.reserve(ops.size());

    // Build in-degree map: count dependencies within the cluster for each op
    llvm::DenseMap<Operation *, unsigned> inDegree;
    for (Operation *op : ops) {
      inDegree[op] = 0;
    }

    // Count in-degrees (only from ops within the cluster)
    for (Operation *op : ops) {
      for (Value operand : op->getOperands()) {
        Operation *defOp = operand.getDefiningOp();
        if (defOp && ops.contains(defOp)) {
          ++inDegree[op];
        }
      }
    }

    // Initialize worklist with ops that have no dependencies within the cluster
    SmallVector<Operation *> worklist;
    std::copy_if(ops.begin(), ops.end(), std::back_inserter(worklist),
                 [&inDegree](Operation *op) { return inDegree[op] == 0; });

    // Process worklist
    while (!worklist.empty()) {
      Operation *op = worklist.pop_back_val();
      sortedOps.push_back(op);

      // Decrease in-degree of users within the cluster
      for (Value result : op->getResults()) {
        for (Operation *user : result.getUsers()) {
          if (ops.contains(user)) {
            --inDegree[user];
            if (inDegree[user] == 0) {
              worklist.push_back(user);
            }
          }
        }
      }
    }

    // Check for cycle: if not all ops are sorted, there's a cycle
    if (sortedOps.size() != ops.size()) {
      LLVM_DEBUG(llvm::dbgs() << "[Fusion] Cycle detected in fusion cluster, "
                              << "sorted " << sortedOps.size() << " of " << ops.size() << " ops\n");
      return {};  // Return empty vector to indicate failure
    }

    return sortedOps;
  }
};

//===----------------------------------------------------------------------===//
// Pass definition
//===----------------------------------------------------------------------===//

}  // namespace

namespace mlir {

#define GEN_PASS_DEF_OUTLINESTABLEHLOFUSIONREGIONS
#include "mopt/Fusion/Passes.h.inc"

struct OutlineStablehloFusionRegionsPass
    : public impl::OutlineStablehloFusionRegionsBase<OutlineStablehloFusionRegionsPass> {
  using OutlineStablehloFusionRegionsBase::OutlineStablehloFusionRegionsBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();

    LLVM_DEBUG(llvm::dbgs() << "[Fusion] Running outline-stablehlo-fusion-regions pass\n");

    // Pre-clean: remove dead Torch symbolic-shape modeling ops that only exist
    // to carry metadata and would otherwise force multi-result clusters.
    module.walk([&](FuncOp funcOp) { cleanupDeadTorchSymbolicShapeBinds(funcOp); });

    // Use configurable StableHLO fusion strategy.
    // Defaults keep existing behavior; options are provided via Passes.td.
    mopt::StablehloFusionStrategy strategy(allowDotGeneral, allowDynamicShape);

    // Step 1: Build fusion clusters using the strategy
    FusionClusterBuilder clusterBuilder(strategy);
    SmallVector<FusionCluster, 4> clusters = clusterBuilder.buildClusters(module);

    if (clusters.empty()) {
      return;
    }

    LLVM_DEBUG(llvm::dbgs() << "[Fusion] Found " << clusters.size() << " fusion clusters\n");

    // Step 2: Outline each cluster and replace with call
    FunctionOutliner outliner(module);
    llvm::StringMap<unsigned> funcClusterIndices;
    unsigned totalOutlined = 0;

    for (auto &cluster : clusters) {
      unsigned clusterIndex = funcClusterIndices[cluster.parentFuncName]++;
      FuncOp outlinedFunc = outliner.outline(cluster, clusterIndex, cluster.parentFuncName);
      if (outlinedFunc) {
        outliner.replaceWithCall(cluster, outlinedFunc);
        ++totalOutlined;
      } else {
        LLVM_DEBUG(llvm::dbgs() << "[Fusion] Failed to create outlined function\n");
      }
    }

    LLVM_DEBUG(llvm::dbgs() << "[Fusion] Pass complete, outlined " << totalOutlined << " clusters\n");
  }
};

std::unique_ptr<Pass> createOutlineStablehloFusionRegionsPass() {
  return std::make_unique<OutlineStablehloFusionRegionsPass>();
}

}  // namespace mlir
