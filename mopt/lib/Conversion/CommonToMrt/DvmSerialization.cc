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

#include "mopt/Conversion/CommonToMrt/DvmSerialization.h"
#include "mopt/Dialect/Dvm/DvmDialect.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

namespace {

// NOTE: For llvm::SmallVector<T, N>, the template parameter `N` is the *inline
// capacity* (how many elements can be stored in-place without heap allocation).
constexpr unsigned kInlineCapInstInputs = 4;
constexpr unsigned kInlineCapInstAttrs = 4;
constexpr unsigned kInlineCapInputIndices = 8;
constexpr unsigned kInlineCapOutputIndices = 8;
constexpr unsigned kInlineCapInsts = 32;

struct InstDesc {
  llvm::StringRef op;
  int32_t idx;
  llvm::SmallVector<int32_t, kInlineCapInstInputs> inputs;
  // Small and fixed set of attrs -> store as vector to keep stable order in JSON output.
  llvm::SmallVector<std::pair<llvm::StringRef, llvm::json::Value>, kInlineCapInstAttrs> attrs;
};

// Use MLIR's print capability to stringify element types (e.g., f16/bf16/f32/i32).
std::string elementTypeToDvmDtype(mlir::Type elementType) {
  std::string s;
  llvm::raw_string_ostream os(s);
  elementType.print(os);
  os.flush();
  return s;
}

llvm::StringRef dvmUnaryTypeToJson(mlir::dvm::UnaryOpType opType) { return mlir::dvm::stringifyUnaryOpType(opType); }

llvm::StringRef dvmBinaryTypeToJson(mlir::dvm::BinaryOpType opType) { return mlir::dvm::stringifyBinaryOpType(opType); }

llvm::StringRef dvmReduceTypeToJson(mlir::dvm::ReduceOpType opType) { return mlir::dvm::stringifyReduceOpType(opType); }

// Convert DVM's DType enum to an MLIR element type token string (e.g., f16/i32).
std::string dvmDTypeToMlirDtypeToken(mlir::dvm::DType dtype, mlir::MLIRContext *ctx) {
  mlir::Type ty;
  switch (dtype) {
    case mlir::dvm::DType::Bool:
      ty = mlir::IntegerType::get(ctx, 1);
      break;
    case mlir::dvm::DType::Float16:
      ty = mlir::Float16Type::get(ctx);
      break;
    case mlir::dvm::DType::BFloat16:
      ty = mlir::BFloat16Type::get(ctx);
      break;
    case mlir::dvm::DType::Float32:
      ty = mlir::Float32Type::get(ctx);
      break;
    case mlir::dvm::DType::Int32:
      ty = mlir::IntegerType::get(ctx, 32);
      break;
    case mlir::dvm::DType::Int64:
      ty = mlir::IntegerType::get(ctx, 64);
      break;
  }
  return elementTypeToDvmDtype(ty);
}

// Serializer for DVM operations to JSON format
class DvmOpSerializer {
 public:
  explicit DvmOpSerializer(mlir::func::FuncOp funcOp, llvm::SmallVectorImpl<mlir::Value> &callInputsOut)
      : funcOp(funcOp), callInputsOut(callInputsOut) {
    callInputsOut.clear();
  }

  mlir::LogicalResult dispatch(mlir::Operation &op);
  mlir::LogicalResult finalizeOutputIndicesFromReturn(mlir::func::ReturnOp returnOp);

  const llvm::SmallVector<int32_t, kInlineCapInputIndices> &getInputIndices() const { return inputIndices; }
  const llvm::SmallVector<int32_t, kInlineCapOutputIndices> &getOutputIndices() const { return outputIndices; }
  const llvm::SmallVector<InstDesc, kInlineCapInsts> &getInstructions() const { return insts; }

 private:
  mlir::func::FuncOp funcOp;
  llvm::DenseMap<mlir::Value, int32_t> valueToIdx;
  llvm::SmallVector<int32_t, kInlineCapInputIndices> inputIndices;
  llvm::SmallVector<int32_t, kInlineCapOutputIndices> outputIndices;
  llvm::SmallVector<InstDesc, kInlineCapInsts> insts;
  int32_t nextIdx = 0;
  llvm::SmallVectorImpl<mlir::Value> &callInputsOut;

  mlir::FailureOr<int32_t> requireValueIdx(mlir::Value v, llvm::StringRef what) {
    auto it = valueToIdx.find(v);
    if (it == valueToIdx.end()) {
      return funcOp.emitError("dvm-serialization: missing value idx for ")
             << what << " (SSA value not produced by a prior DVM op)";
    }
    return it->second;
  }

  int32_t assignIdxForResult(mlir::Value result) {
    int32_t idx = nextIdx++;
    valueToIdx[result] = idx;
    return idx;
  }

  // Helper class to build instruction descriptors with fluent API.
  class InstBuilder {
    DvmOpSerializer &serializer;
    InstDesc inst;
    bool failed = false;

   public:
    InstBuilder(DvmOpSerializer &s, llvm::StringRef opName, mlir::Value result) : serializer(s) {
      inst.idx = serializer.assignIdxForResult(result);
      inst.op = opName;
    }

    InstBuilder &addInput(mlir::Value v, llvm::StringRef what) {
      if (failed) return *this;
      auto idx = serializer.requireValueIdx(v, what);
      if (mlir::failed(idx)) {
        failed = true;
        return *this;
      }
      inst.inputs.push_back(*idx);
      return *this;
    }

    InstBuilder &addAttr(llvm::StringRef key, llvm::json::Value value) {
      inst.attrs.push_back({key, std::move(value)});
      return *this;
    }

    mlir::LogicalResult finalize() {
      if (failed) return mlir::failure();
      serializer.insts.push_back(std::move(inst));
      return mlir::success();
    }
  };

  mlir::LogicalResult handleLoad(mlir::dvm::LoadOp loadOp);
  mlir::LogicalResult handleStore(mlir::dvm::StoreOp storeOp);
  mlir::LogicalResult handleUnary(mlir::dvm::UnaryOp unaryOp);
  mlir::LogicalResult handleBinary(mlir::dvm::BinaryOp binaryOp);
  mlir::LogicalResult handleMatmul(mlir::dvm::MatMulOp matmulOp);
  mlir::LogicalResult handleReduce(mlir::dvm::ReduceOp reduceOp);
  mlir::LogicalResult handleCast(mlir::dvm::CastOp castOp);
  mlir::LogicalResult handleReshape(mlir::dvm::ReshapeOp reshapeOp);
  mlir::LogicalResult handleBroadcast(mlir::dvm::BroadcastOp bcastOp);
  mlir::LogicalResult handleSelect(mlir::dvm::SelectOp selectOp);
};

mlir::LogicalResult DvmOpSerializer::handleLoad(mlir::dvm::LoadOp loadOp) {
  int32_t idx = assignIdxForResult(loadOp.getResult());
  inputIndices.push_back(idx);

  if (!mlir::isa<mlir::BlockArgument>(loadOp.getInput())) {
    return loadOp.emitError("dvm-serialization: dvm.load input must be a block argument");
  }
  callInputsOut.push_back(loadOp.getInput());

  mlir::Type ty = loadOp.getResult().getType();
  auto shaped = mlir::dyn_cast<mlir::ShapedType>(ty);
  if (!shaped || !shaped.hasRank()) {
    return loadOp.emitError("dvm-serialization: dvm.load result must be a ranked tensor");
  }

  InstDesc inst;
  inst.op = "load";
  inst.idx = idx;
  inst.attrs.push_back({"dtype", elementTypeToDvmDtype(shaped.getElementType())});
  insts.push_back(std::move(inst));
  return mlir::success();
}

mlir::LogicalResult DvmOpSerializer::handleStore(mlir::dvm::StoreOp storeOp) {
  int32_t idx = assignIdxForResult(storeOp.getResult());

  auto srcIdx = requireValueIdx(storeOp.getInput(), "dvm.store input");
  if (mlir::failed(srcIdx)) return mlir::failure();

  InstDesc inst;
  inst.op = "store";
  inst.idx = idx;
  inst.inputs.push_back(*srcIdx);
  insts.push_back(std::move(inst));
  return mlir::success();
}

mlir::LogicalResult DvmOpSerializer::finalizeOutputIndicesFromReturn(mlir::func::ReturnOp returnOp) {
  outputIndices.clear();
  outputIndices.reserve(returnOp.getNumOperands());
  for (mlir::Value ret : returnOp.getOperands()) {
    auto idx = requireValueIdx(ret, "func.return operand");
    if (mlir::failed(idx)) return mlir::failure();
    outputIndices.push_back(*idx);
  }
  return mlir::success();
}

mlir::LogicalResult DvmOpSerializer::handleUnary(mlir::dvm::UnaryOp unaryOp) {
  return InstBuilder(*this, "unary", unaryOp.getResult())
    .addInput(unaryOp.getInput(), "dvm.unary input")
    .addAttr("type", dvmUnaryTypeToJson(unaryOp.getOpType()))
    .finalize();
}

mlir::LogicalResult DvmOpSerializer::handleBinary(mlir::dvm::BinaryOp binaryOp) {
  return InstBuilder(*this, "binary", binaryOp.getResult())
    .addInput(binaryOp.getLhs(), "dvm.binary lhs")
    .addInput(binaryOp.getRhs(), "dvm.binary rhs")
    .addAttr("type", dvmBinaryTypeToJson(binaryOp.getOpType()))
    .finalize();
}

mlir::LogicalResult DvmOpSerializer::handleMatmul(mlir::dvm::MatMulOp matmulOp) {
  if (matmulOp.getBias()) {
    return matmulOp.emitError("dvm-serialization: dvm.matmul with bias is not supported by OpDvmCall");
  }
  return InstBuilder(*this, "matmul", matmulOp.getResult())
    .addInput(matmulOp.getLhs(), "dvm.matmul lhs")
    .addInput(matmulOp.getRhs(), "dvm.matmul rhs")
    .addAttr("trans_a", matmulOp.getTransA())
    .addAttr("trans_b", matmulOp.getTransB())
    .finalize();
}

mlir::LogicalResult DvmOpSerializer::handleReduce(mlir::dvm::ReduceOp reduceOp) {
  llvm::json::Array axes;
  std::transform(reduceOp.getDims().begin(), reduceOp.getDims().end(), std::back_inserter(axes),
                 [](mlir::Attribute dimAttr) { return mlir::cast<mlir::IntegerAttr>(dimAttr).getInt(); });
  return InstBuilder(*this, "reduce", reduceOp.getResult())
    .addInput(reduceOp.getInput(), "dvm.reduce input")
    .addAttr("type", dvmReduceTypeToJson(reduceOp.getOpType()))
    .addAttr("axes", std::move(axes))
    .addAttr("keepdims", reduceOp.getKeepdims())
    .finalize();
}

mlir::LogicalResult DvmOpSerializer::handleCast(mlir::dvm::CastOp castOp) {
  return InstBuilder(*this, "cast", castOp.getResult())
    .addInput(castOp.getInput(), "dvm.cast input")
    .addAttr("dtype", dvmDTypeToMlirDtypeToken(castOp.getType(), castOp.getContext()))
    .finalize();
}

mlir::LogicalResult DvmOpSerializer::handleReshape(mlir::dvm::ReshapeOp reshapeOp) {
  return InstBuilder(*this, "reshape", reshapeOp.getResult())
    .addInput(reshapeOp.getInput(), "dvm.reshape input")
    .finalize();
}

mlir::LogicalResult DvmOpSerializer::handleBroadcast(mlir::dvm::BroadcastOp bcastOp) {
  return InstBuilder(*this, "broadcast", bcastOp.getResult())
    .addInput(bcastOp.getInput(), "dvm.broadcast input")
    .finalize();
}

mlir::LogicalResult DvmOpSerializer::handleSelect(mlir::dvm::SelectOp selectOp) {
  return InstBuilder(*this, "select", selectOp.getResult())
    .addInput(selectOp.getCond(), "dvm.select cond")
    .addInput(selectOp.getLhs(), "dvm.select lhs")
    .addInput(selectOp.getRhs(), "dvm.select rhs")
    .finalize();
}

mlir::LogicalResult DvmOpSerializer::dispatch(mlir::Operation &op) {
  return llvm::TypeSwitch<mlir::Operation *, mlir::LogicalResult>(&op)
    .Case<mlir::dvm::LoadOp>([&](auto v) { return handleLoad(v); })
    .Case<mlir::dvm::StoreOp>([&](auto v) { return handleStore(v); })
    .Case<mlir::dvm::UnaryOp>([&](auto v) { return handleUnary(v); })
    .Case<mlir::dvm::BinaryOp>([&](auto v) { return handleBinary(v); })
    .Case<mlir::dvm::MatMulOp>([&](auto v) { return handleMatmul(v); })
    .Case<mlir::dvm::ReduceOp>([&](auto v) { return handleReduce(v); })
    .Case<mlir::dvm::CastOp>([&](auto v) { return handleCast(v); })
    .Case<mlir::dvm::ReshapeOp>([&](auto v) { return handleReshape(v); })
    .Case<mlir::dvm::BroadcastOp>([&](auto v) { return handleBroadcast(v); })
    .Case<mlir::dvm::SelectOp>([&](auto v) { return handleSelect(v); })
    .Default([&](mlir::Operation *unknown) -> mlir::LogicalResult {
      if (unknown->getDialect() &&
          unknown->getDialect()->getNamespace() == mlir::dvm::DvmDialect::getDialectNamespace()) {
        return unknown->emitError("dvm-serialization: unsupported DVM op in function: ") << unknown->getName();
      }
      return unknown->emitError("dvm-serialization: non-DVM op in DVM function: ") << unknown->getName();
    });
}

// Validates function structure for DVM serialization
static mlir::LogicalResult validateFuncForSerialization(mlir::func::FuncOp funcOp, mlir::Block *&blockOut,
                                                        mlir::func::ReturnOp &returnOpOut) {
  if (funcOp.isExternal()) return mlir::success();

  if (!funcOp.getBody().hasOneBlock()) {
    return funcOp.emitError("dvm-serialization: expected single-block func.func");
  }

  blockOut = &funcOp.front();
  returnOpOut = mlir::dyn_cast<mlir::func::ReturnOp>(blockOut->getTerminator());
  if (!returnOpOut) {
    return funcOp.emitError("dvm-serialization: missing func.return terminator");
  }

  return mlir::success();
}

// Checks if block contains any DVM operations
static bool containsDvmOps(mlir::Block &block) {
  auto range = block.without_terminator();
  return std::any_of(range.begin(), range.end(), [](mlir::Operation &op) {
    return op.getDialect() && op.getDialect()->getNamespace() == mlir::dvm::DvmDialect::getDialectNamespace();
  });
}

// Validates return operands come from dvm.store
static mlir::LogicalResult validateReturnOperands(mlir::func::ReturnOp returnOp) {
  for (mlir::Value ret : returnOp.getOperands()) {
    auto *def = ret.getDefiningOp();
    if (!def || !mlir::isa<mlir::dvm::StoreOp>(def)) {
      return returnOp.emitError("dvm-serialization: func.return must return dvm.store results");
    }
  }
  return mlir::success();
}

// Serializes instructions to JSON format
static void serializeToJson(const DvmOpSerializer &serializer, llvm::StringRef kernelType, unsigned indent,
                            std::string &jsonOut) {
  std::string json;
  llvm::raw_string_ostream os(json);
  llvm::json::OStream j(os, indent);

  j.object([&] {
    j.attribute("version", 1);
    j.attribute("kernel_type", kernelType);
    j.attributeArray("instructions", [&] {
      for (const auto &inst : serializer.getInstructions()) {
        j.object([&] {
          j.attribute("op", inst.op);
          j.attribute("idx", inst.idx);
          j.attributeArray("inputs", [&] {
            for (int32_t in : inst.inputs) j.value(in);
          });
          j.attributeObject("attrs", [&] {
            for (const auto &kv : inst.attrs) {
              j.attribute(kv.first, kv.second);
            }
          });
        });
      }
    });
    j.attributeArray("input_indices", [&] {
      for (int32_t idx : serializer.getInputIndices()) j.value(idx);
    });
    j.attributeArray("output_indices", [&] {
      for (int32_t idx : serializer.getOutputIndices()) j.value(idx);
    });
  });
  os.flush();
  jsonOut = std::move(json);
}

}  // namespace

namespace mlir {
namespace dvm {

mlir::LogicalResult serializeDvmFuncToJson(mlir::func::FuncOp funcOp, llvm::StringRef kernelType, unsigned indent,
                                           std::string &jsonOut, llvm::SmallVectorImpl<mlir::Value> &callInputsOut) {
  mlir::Block *block;
  mlir::func::ReturnOp returnOp;

  if (mlir::failed(validateFuncForSerialization(funcOp, block, returnOp))) {
    return mlir::failure();
  }

  if (!block || funcOp.isExternal()) return mlir::success();

  if (!containsDvmOps(*block)) return mlir::success();

  DvmOpSerializer serializer(funcOp, callInputsOut);
  for (mlir::Operation &op : block->without_terminator()) {
    if (mlir::failed(serializer.dispatch(op))) return mlir::failure();
  }

  if (mlir::failed(validateReturnOperands(returnOp))) {
    return mlir::failure();
  }

  if (mlir::failed(serializer.finalizeOutputIndicesFromReturn(returnOp))) {
    return mlir::failure();
  }

  serializeToJson(serializer, kernelType, indent, jsonOut);
  return mlir::success();
}

}  // namespace dvm
}  // namespace mlir
