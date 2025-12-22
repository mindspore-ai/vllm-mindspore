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
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionDialect.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionOps.h"

namespace mlir {
namespace {

// A defensive upper bound for chasing no-op cast chains when attaching symbolic-shape metadata.
//
// Why this exists:
// - Some pipelines can materialize long chains of `tensor.cast` / `unrealized_conversion_cast` during type
//   reconciliation.
// - Canonicalization may later erase these casts; we prefer to attach metadata on the "real" producer.
// - The cast chain is not expected to be long in normal cases; this cap prevents pathological IR (or accidental
//   cycles) from causing excessive work.
//
// Note: Hitting this cap only affects optimization/metadata visibility, not correctness.
static constexpr int kMaxNoOpCastChaseDepth = 8;

static void setSymbolicShapeAttrOnProducer(Value builtinTensor, DictionaryAttr sigAttr) {
  if (!builtinTensor) return;

  // Attach the signature to the defining op, and also try to push it through
  // common no-op casts so downstream consumers that look at the "real" producer
  // (e.g. stablehlo.dot) can see it.
  //
  // This is important because some conversion patterns may materialize
  // tensor.cast / unrealized_conversion_cast values for type reconciliation,
  // and later canonicalization can erase these casts. If we only annotate the
  // immediate cast op, the attribute may be lost before CHLO->StableHLO
  // legalization consults it.
  Value cur = builtinTensor;
  for (int i = 0; i < kMaxNoOpCastChaseDepth; ++i) {
    Operation *defOp = cur.getDefiningOp();
    if (!defOp) break;
    defOp->setAttr(mopt::fusion::kSymbolicShapeAttrName, sigAttr);

    if (auto castOp = dyn_cast<mlir::tensor::CastOp>(defOp)) {
      cur = castOp.getSource();
      continue;
    }
    if (auto ucc = dyn_cast<mlir::UnrealizedConversionCastOp>(defOp)) {
      if (ucc.getInputs().empty()) break;
      cur = ucc.getInputs().front();
      continue;
    }
    break;
  }
}

}  // namespace
}  // namespace mlir

namespace mlir {

#define GEN_PASS_DEF_PROPAGATETORCHSYMBOLICSHAPE
#include "mopt/Fusion/Passes.h.inc"

struct PropagateTorchSymbolicShapePass : public impl::PropagateTorchSymbolicShapeBase<PropagateTorchSymbolicShapePass> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();

    // Map torch !torch.vtensor values to a canonical signature.
    llvm::DenseMap<Value, DictionaryAttr> torchValueToSig;
    bool failed = false;

    func.walk([&](torch::Torch::BindSymbolicShapeOp op) {
      if (failed) return;
      Value tensor = op.getOperand();
      DictionaryAttr sigAttr = mopt::fusion::buildSymbolicShapeSignature(op);
      if (!sigAttr) {
        // buildSymbolicShapeSignature already emitted a diagnostic.
        failed = true;
        return;
      }

      auto it = torchValueToSig.find(tensor);
      if (it == torchValueToSig.end()) {
        torchValueToSig.try_emplace(tensor, sigAttr);
        return;
      }
      // If multiple bindings exist, keep the first if consistent; otherwise
      // treat it as an IR invariant violation and fail the pass.
      if (it->second != sigAttr) {
        op.emitError() << "Conflicting torch.bind_symbolic_shape for the same tensor. existing="
                       << mopt::fusion::symbolicShapeSignatureToString(it->second)
                       << " new=" << mopt::fusion::symbolicShapeSignatureToString(sigAttr);
        failed = true;
        return;
      }
    });

    if (failed) {
      signalPassFailure();
      return;
    }

    if (torchValueToSig.empty()) return;

    // Propagate torch->builtin via torch_c.to_builtin_tensor.
    func.walk([&](torch::TorchConversion::ToBuiltinTensorOp op) {
      auto it = torchValueToSig.find(op.getOperand());
      if (it == torchValueToSig.end()) return;
      op->setAttr(mopt::fusion::kSymbolicShapeAttrName, it->second);
    });

    // Propagate builtin<-torch via torch_c.from_builtin_tensor.
    // If the torch result is bound symbolically, attach the signature to the
    // producer of the builtin tensor operand (e.g. stablehlo.dot).
    func.walk([&](torch::TorchConversion::FromBuiltinTensorOp op) {
      auto it = torchValueToSig.find(op.getResult());
      if (it == torchValueToSig.end()) return;
      setSymbolicShapeAttrOnProducer(op.getOperand(), it->second);
    });
  }
};

std::unique_ptr<Pass> createPropagateTorchSymbolicShapePass() {
  return std::make_unique<PropagateTorchSymbolicShapePass>();
}

}  // namespace mlir
