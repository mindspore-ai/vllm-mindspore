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

#include "mopt/Fusion/FusionStrategy.h"
#include "mopt/Fusion/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "stablehlo/dialect/StablehloOps.h"
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

      // Only fuse if all users of the producer are in the cluster or if it's a constant
      bool allUsersInCluster = true;
      for (Operation *user : defOp->getResult(0).getUsers()) {
        if (user != root && !cluster.ops.contains(user)) {
          allUsersInCluster = false;
          break;
        }
      }

      if (allUsersInCluster || isa<mlir::stablehlo::ConstantOp>(defOp)) {
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

        // Check if user only depends on ops already in the cluster
        bool canFuseUser = true;
        for (Value userOperand : user->getOperands()) {
          Operation *userDefOp = userOperand.getDefiningOp();
          if (userDefOp && !cluster.ops.contains(userDefOp) && userOperand != result) {
            // Has operand from outside the cluster that's not the current result
            canFuseUser = false;
            break;
          }
        }
        if (canFuseUser) {
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
    for (Value input : cluster.inputs) {
      inputTypes.push_back(input.getType());
    }

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
    for (Value output : cluster.outputs) {
      returnValues.push_back(mapper.lookup(output));
    }
    mlir::func::ReturnOp::create(funcBuilder, loc, returnValues);

    return funcOp;
  }

  /// Replace a fusion cluster with a call to the outlined function
  void replaceWithCall(FusionCluster &cluster, FuncOp outlinedFunc) {
    if (cluster.ops.empty() || !outlinedFunc) {
      return;
    }

    // Find the last operation in the cluster (for insertion point)
    Operation *lastOp = nullptr;
    for (Operation *op : cluster.ops) {
      if (!lastOp || lastOp->isBeforeInBlock(op)) {
        lastOp = op;
      }
    }

    if (!lastOp) {
      LLVM_DEBUG(llvm::dbgs() << "[Fusion] Could not find last op in cluster\n");
      return;
    }

    // Create call after the last op
    OpBuilder builder(lastOp);
    builder.setInsertionPointAfter(lastOp);

    auto callOp = mlir::func::CallOp::create(builder, lastOp->getLoc(), outlinedFunc, cluster.inputs);

    // Replace uses of cluster outputs with call results
    llvm::SmallPtrSet<Operation *, 16> clusterOpsSet;
    for (Operation *op : cluster.ops) {
      clusterOpsSet.insert(op);
    }
    for (size_t i = 0; i < cluster.outputs.size(); ++i) {
      cluster.outputs[i].replaceAllUsesExcept(callOp.getResult(i), clusterOpsSet);
    }

    // Erase cluster operations in reverse order (last to first in block)
    SmallVector<Operation *> sortedOps(cluster.ops.begin(), cluster.ops.end());
    std::sort(sortedOps.begin(), sortedOps.end(), [](Operation *a, Operation *b) {
      return b->isBeforeInBlock(a);
    });

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
    for (Operation *op : ops) {
      if (inDegree[op] == 0) {
        worklist.push_back(op);
      }
    }

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

struct OutlineStablehloFusionRegionsPass
    : public PassWrapper<OutlineStablehloFusionRegionsPass, OperationPass<ModuleOp>> {
  // cppcheck-suppress unknownMacro
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OutlineStablehloFusionRegionsPass)

  StringRef getArgument() const final { return "outline-stablehlo-fusion-regions"; }

  StringRef getDescription() const final { return "Outline StableHLO fusion regions into separate functions"; }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::stablehlo::StablehloDialect>();
    registry.insert<mlir::func::FuncDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    LLVM_DEBUG(llvm::dbgs() << "[Fusion] Running outline-stablehlo-fusion-regions pass\n");

    // Use StableHLO fusion strategy
    mopt::StablehloFusionStrategy strategy;

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

}  // namespace

namespace mlir {

std::unique_ptr<Pass> createOutlineStablehloFusionRegionsPass() {
  return std::make_unique<OutlineStablehloFusionRegionsPass>();
}

}  // namespace mlir
