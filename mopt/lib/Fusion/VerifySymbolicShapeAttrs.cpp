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
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"

namespace mlir {

#define GEN_PASS_DEF_VERIFYSYMBOLICSHAPEATTRS
#include "mopt/Fusion/Passes.h.inc"

namespace {

struct VerifySymbolicShapeAttrsPass : public impl::VerifySymbolicShapeAttrsBase<VerifySymbolicShapeAttrsPass> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    MLIRContext *ctx = func.getContext();
    (void)ctx;

    func.walk([&](Operation *op) {
      if (!op->hasAttr(mlir::mopt::fusion::kSymbolicShapeAttrName)) return;

      auto sig = op->getAttr(mlir::mopt::fusion::kSymbolicShapeAttrName);
      auto dict = dyn_cast<DictionaryAttr>(sig);
      if (!dict) {
        op->emitError("mopt.symbolic_shape must be a DictionaryAttr (schema v1)");
        signalPassFailure();
        return;
      }

      auto ver = dict.getAs<StringAttr>("ver");
      auto exprs = dict.getAs<AffineMapAttr>("exprs");
      auto syms = dict.getAs<ArrayAttr>("syms");
      if (!ver || !exprs || !syms) {
        op->emitError("mopt.symbolic_shape missing required keys {ver, exprs, syms}");
        signalPassFailure();
        return;
      }
      if (ver.getValue() != mlir::mopt::fusion::kSymbolicShapeSchemaVersion) {
        op->emitError("mopt.symbolic_shape has unsupported schema version: ") << ver.getValue();
        signalPassFailure();
        return;
      }
      for (Attribute a : syms) {
        if (!isa<StringAttr>(a)) {
          op->emitError("mopt.symbolic_shape.syms must be ArrayAttr<StringAttr>");
          signalPassFailure();
          return;
        }
      }
    });
  }
};

}  // namespace

std::unique_ptr<Pass> createVerifySymbolicShapeAttrsPass() { return std::make_unique<VerifySymbolicShapeAttrsPass>(); }

}  // namespace mlir
