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

#include "mopt/Dialect/Mrt/Transforms/Passes.h"

#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {

#define GEN_PASS_DEF_ANNOTATELINALGFORHACC
#include "mopt/Dialect/Mrt/Transforms/Passes.h.inc"

namespace {

/// Pass to annotate functions containing Linalg operations with hacc attributes.
/// This prepares the MLIR for BiSheng IR compilation by marking entry points.
struct AnnotateLinalgForHaccPass : public impl::AnnotateLinalgForHaccBase<AnnotateLinalgForHaccPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *context = &getContext();

    // Walk through all functions in the module
    module.walk([&](func::FuncOp funcOp) {
      // Skip if already annotated
      if (funcOp->hasAttr("hacc.entry")) {
        return;
      }

      // Check if function contains any Linalg operations
      bool hasLinalgOp = false;
      funcOp.walk([&](Operation *op) {
        if (isa<linalg::LinalgOp>(op)) {
          hasLinalgOp = true;
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });

      // If function contains Linalg operations, add hacc attributes
      if (hasLinalgOp) {
        // Add hacc.entry attribute (UnitAttr)
        funcOp->setAttr("hacc.entry", UnitAttr::get(context));

        // Parse hacc.function_kind as a typed attribute
        Attribute functionKindAttr = parseAttribute("#hacc.function_kind<HOST>", context);
        if (!functionKindAttr) {
          funcOp.emitError("Failed to parse hacc.function_kind attribute");
          return;
        }

        funcOp->setAttr("hacc.function_kind", functionKindAttr);
      }
    });
  }
};

}  // namespace

std::unique_ptr<Pass> createAnnotateLinalgForHaccPass() { return std::make_unique<AnnotateLinalgForHaccPass>(); }

}  // namespace mlir
