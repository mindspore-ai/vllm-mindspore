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

#include "mopt/Fusion/SymbolicShape.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::mopt::fusion {

static std::string valueToDebugString(Value v) {
  std::string s;
  llvm::raw_string_ostream os(s);
  v.print(os);
  os.flush();
  return s;
}

DictionaryAttr buildSymbolicShapeSignature(torch::Torch::BindSymbolicShapeOp op) {
  MLIRContext *ctx = op.getContext();

  // exprs: AffineMapAttr
  auto exprs = op.getShapeExpressions();

  // syms: ArrayAttr<StringAttr>
  SmallVector<Attribute> syms;
  syms.reserve(op.getShapeSymbols().size());
  for (Value sym : op.getShapeSymbols()) {
    if (auto symOp = sym.getDefiningOp<torch::Torch::SymbolicIntOp>()) {
      syms.push_back(StringAttr::get(ctx, symOp.getSymbolName()));
    } else {
      // This is treated as IR invariant violation: bind_symbolic_shape should only
      // reference symbols produced by torch.symbolic_int.
      op.emitError("bind_symbolic_shape shape symbol must be produced by torch.symbolic_int, got: ")
        << valueToDebugString(sym);
      return nullptr;
    }
  }

  NamedAttrList nal;
  nal.set("ver", StringAttr::get(ctx, kSymbolicShapeSchemaVersion));
  nal.set("exprs", exprs);
  nal.set("syms", ArrayAttr::get(ctx, syms));
  return DictionaryAttr::get(ctx, nal);
}

std::string symbolicShapeSignatureToString(DictionaryAttr sig) {
  if (!sig) return "<null>";
  std::string s;
  llvm::raw_string_ostream os(s);
  sig.print(os);
  os.flush();
  return s;
}

DictionaryAttr getSymbolicShapeSignature(Value v) {
  if (!v) return nullptr;
  Operation *defOp = v.getDefiningOp();
  if (!defOp) return nullptr;
  return defOp->getAttrOfType<DictionaryAttr>(kSymbolicShapeAttrName);
}

}  // namespace mlir::mopt::fusion
