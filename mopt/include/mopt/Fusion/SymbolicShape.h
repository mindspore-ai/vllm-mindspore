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

#ifndef MOPT_FUSION_SYMBOLIC_SHAPE_H_
#define MOPT_FUSION_SYMBOLIC_SHAPE_H_

#include <string>

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/StringRef.h"

// NOTE: This header is used by fusion passes that depend on torch-mlir ops.
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"

namespace mlir::mopt::fusion {

/// Attribute name used across the fusion pipeline to carry symbolic-shape equivalence metadata.
///
/// This attribute is intended to be attached to "real" tensor producer ops (e.g. stablehlo.dot),
/// not ephemeral casts, so that downstream shape/broadcast canonicalization can consult it.
inline constexpr llvm::StringRef kSymbolicShapeAttrName = "mopt.symbolic_shape";

/// Schema version for the attribute payload (for forward compatibility).
inline constexpr llvm::StringRef kSymbolicShapeSchemaVersion = "v1";

/// Build a stable, structurally comparable signature from a `torch.bind_symbolic_shape`.
///
/// We intentionally use a structured attribute (DictionaryAttr) instead of a string to:
/// - make equality comparisons robust (no string formatting fragility)
/// - allow future schema evolution (versioned payload)
///
/// Schema (DictionaryAttr):
/// - "ver": StringAttr (currently "v1")
/// - "exprs": AffineMapAttr (from bind_symbolic_shape.shape_expressions)
/// - "syms": ArrayAttr<StringAttr> (symbol names; must come from `torch.symbolic_int`)
DictionaryAttr buildSymbolicShapeSignature(torch::Torch::BindSymbolicShapeOp op);

/// Human-friendly debug string for a signature produced by `buildSymbolicShapeSignature`.
std::string symbolicShapeSignatureToString(DictionaryAttr sig);

/// Read symbolic-shape signature attached to the defining op of a value (if any).
DictionaryAttr getSymbolicShapeSignature(Value v);

/// Attach signature on an op (or value's defining op) as `mopt.symbolic_shape`.
inline void setSymbolicShapeSignature(Operation *op, DictionaryAttr sig) {
  if (!op || !sig) return;
  op->setAttr(kSymbolicShapeAttrName, sig);
}

}  // namespace mlir::mopt::fusion

#endif  // MOPT_FUSION_SYMBOLIC_SHAPE_H_
