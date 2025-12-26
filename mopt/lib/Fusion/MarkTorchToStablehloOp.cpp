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

#include <cstdlib>

#include "mopt/Fusion/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/StringExtras.h"

namespace mlir {

#define GEN_PASS_DEF_MARKTORCHTOSTABLEHLOOP
#include "mopt/Fusion/Passes.h.inc"

namespace {

class MarkTorchToStablehloOp : public impl::MarkTorchToStablehloOpBase<MarkTorchToStablehloOp> {
 public:
  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Parse environment variable
    llvm::StringSet<> whitelist;
    if (const char *env_p = std::getenv("MOPT_TORCH_TO_STABLEHLO_WHITELIST")) {
      StringRef envStr(env_p);
      SmallVector<StringRef> parts;
      envStr.split(parts, ',');
      for (StringRef part : parts) {
        StringRef trimmed = part.trim();
        if (!trimmed.empty()) {
          whitelist.insert(trimmed);
        }
      }
    }

    bool matchAll = whitelist.contains("all");

    // Traverse all operations
    module.walk([&](Operation *op) {
      // Only consider ops in the "torch" dialect.
      // This is safer than checking starts_with("torch.")
      if (op->getDialect()->getNamespace() != "torch") {
        return;
      }

      bool shouldConvert = false;

      // If "all" is present, mark all torch ops
      if (matchAll) {
        shouldConvert = true;
      }

      // Check environment whitelist
      if (!shouldConvert && whitelist.contains(op->getName().getStringRef())) {
        shouldConvert = true;
      }

      // Check custom rules (currently empty)
      // if (isCustomMatched(op)) { shouldConvert = true; }

      // Add attribute if matched
      if (shouldConvert) {
        op->setAttr("mopt.torch_to_stablehlo", UnitAttr::get(&getContext()));
      }
    });
  }
};

}  // namespace

std::unique_ptr<Pass> createMarkTorchToStablehloOpPass() { return std::make_unique<MarkTorchToStablehloOp>(); }

}  // namespace mlir
