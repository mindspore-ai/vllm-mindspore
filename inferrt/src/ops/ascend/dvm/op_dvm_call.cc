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

#include "ops/ascend/dvm/op_dvm_call.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <initializer_list>
#include <memory>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "common/logger.h"
#include "common/throw.h"
#include "ir/value/value.h"
#include "ir/tensor/tensor.h"
#include "ops/ascend/dvm/dvm_op.h"
#include "ops/ascend/dvm/dvm_payload.h"

namespace mrt {
namespace ops {

namespace {

constexpr size_t kOpCodeCount = static_cast<size_t>(DvmOpCode::kOpCodeCount);

inline dvm::KernelType KernelTypeFromString(std::string_view kernel_type) {
  static const std::unordered_map<std::string_view, dvm::KernelType> kKernelTypeMap = {
    {"static_shape", dvm::kStaticShape},
    {"dyn_shape", dvm::kDynShape},
    {"static_parallel", dvm::kStaticParallel},
    {"static_mix", dvm::kStaticMix},
    {"dyn_mix", dvm::kDynMix},
    {"static_stages", dvm::kStaticStages},
    {"eager", dvm::kEager},
  };
  const auto it = kKernelTypeMap.find(kernel_type);
  if (it == kKernelTypeMap.end()) {
    MRT_THROW("Unsupported kernel_type in DVM payload: ", kernel_type);
  }
  return it->second;
}

struct BuildCtx {
  dvm::Kernel *k{nullptr};
  const std::vector<dvm::ShapeRef *> *inputShapeRefs{nullptr};
  const dvm::ShapeRef *outputShapeRef{nullptr};
  std::vector<dvm::NDObject *> *values{nullptr};
  size_t *inputLoadCount{nullptr};
  std::vector<dvm::NDObject *> *inputObjs{nullptr};
  std::vector<dvm::NDObject *> *outputObjs{nullptr};
  std::vector<dvm::ShapeRef> *reduceAxesRefs{nullptr};  // stable ShapeRef storage for Reduce dims
  bool ok{true};
};

inline bool IsValidValueRef(const std::vector<dvm::NDObject *> &values, int32_t idx) {
  return idx >= 0 && static_cast<size_t>(idx) < values.size() && values[static_cast<size_t>(idx)] != nullptr;
}

inline bool IsValidResultIdx(const std::vector<dvm::NDObject *> &values, int32_t idx) {
  return idx >= 0 && static_cast<size_t>(idx) < values.size();
}

inline void Fail(BuildCtx *ctx, const char *msg) {
  ctx->ok = false;
  LOG_ERROR << msg;
}

inline bool RequireValidResultIdx(BuildCtx *ctx, int32_t idx, const char *error_msg) {
  if (!IsValidResultIdx(*ctx->values, idx)) {
    Fail(ctx, error_msg);
    return false;
  }
  return true;
}

inline bool RequireValidValueRefs(BuildCtx *ctx, std::initializer_list<int32_t> idxs, const char *error_msg) {
  if (std::any_of(idxs.begin(), idxs.end(), [ctx](int32_t idx) { return !IsValidValueRef(*ctx->values, idx); })) {
    Fail(ctx, error_msg);
    return false;
  }
  return true;
}

inline bool CheckResult(BuildCtx *ctx, dvm::NDObject *obj, const char *op_name) {
  if (obj == nullptr) {
    ctx->ok = false;
    LOG_ERROR << op_name << " returned null pointer (op failed)";
    return false;
  }
  return true;
}

using BuildHandler = bool (*)(const DvmInstruction &, BuildCtx *);

bool HandleLoad(const DvmInstruction &inst, BuildCtx *ctx) {
  if (!RequireValidResultIdx(ctx, inst.result_idx, "Invalid result_idx for Load instruction")) return false;
  if (*ctx->inputLoadCount >= ctx->inputShapeRefs->size()) {
    Fail(ctx, "Too many Load instructions for available inputs");
    return false;
  }
  const auto dtype = static_cast<dvm::DType>(inst.aux_int);
  auto *obj = ctx->k->Load(nullptr, (*ctx->inputShapeRefs)[*ctx->inputLoadCount], dtype);
  if (!CheckResult(ctx, obj, "Load")) return false;
  (*ctx->values)[static_cast<size_t>(inst.result_idx)] = obj;
  ctx->inputObjs->push_back(obj);
  (*ctx->inputLoadCount)++;
  return true;
}

bool HandleStore(const DvmInstruction &inst, BuildCtx *ctx) {
  if (!RequireValidResultIdx(ctx, inst.result_idx, "Invalid result_idx for Store instruction")) return false;
  if (!RequireValidValueRefs(ctx, {inst.operand_idxs[0]}, "Invalid operand for Store instruction")) return false;
  auto *src = (*ctx->values)[static_cast<size_t>(inst.operand_idxs[0])];
  auto *obj = ctx->k->Store(nullptr, src);
  if (!CheckResult(ctx, obj, "Store")) return false;
  (*ctx->values)[static_cast<size_t>(inst.result_idx)] = obj;
  ctx->outputObjs->push_back(obj);
  return true;
}

bool HandleBinary(const DvmInstruction &inst, BuildCtx *ctx) {
  if (!RequireValidResultIdx(ctx, inst.result_idx, "Invalid result_idx for Binary instruction")) return false;
  if (!RequireValidValueRefs(ctx, {inst.operand_idxs[0], inst.operand_idxs[1]},
                             "Invalid operands for Binary instruction"))
    return false;
  auto *lhs = (*ctx->values)[static_cast<size_t>(inst.operand_idxs[0])];
  auto *rhs = (*ctx->values)[static_cast<size_t>(inst.operand_idxs[1])];
  auto *obj = ctx->k->Binary(inst.aux_int, lhs, rhs);
  if (!CheckResult(ctx, obj, "Binary")) return false;
  (*ctx->values)[static_cast<size_t>(inst.result_idx)] = obj;
  return true;
}

bool HandleUnary(const DvmInstruction &inst, BuildCtx *ctx) {
  if (!RequireValidResultIdx(ctx, inst.result_idx, "Invalid result_idx for Unary instruction")) return false;
  if (!RequireValidValueRefs(ctx, {inst.operand_idxs[0]}, "Invalid operand for Unary instruction")) return false;
  auto *src = (*ctx->values)[static_cast<size_t>(inst.operand_idxs[0])];
  auto *obj = ctx->k->Unary(inst.aux_int, src);
  if (!CheckResult(ctx, obj, "Unary")) return false;
  (*ctx->values)[static_cast<size_t>(inst.result_idx)] = obj;
  return true;
}

bool HandleMatMul(const DvmInstruction &inst, BuildCtx *ctx) {
  if (!RequireValidResultIdx(ctx, inst.result_idx, "Invalid result_idx for MatMul instruction")) return false;
  if (!RequireValidValueRefs(ctx, {inst.operand_idxs[0], inst.operand_idxs[1]},
                             "Invalid operands for MatMul instruction"))
    return false;
  auto *lhs = (*ctx->values)[static_cast<size_t>(inst.operand_idxs[0])];
  auto *rhs = (*ctx->values)[static_cast<size_t>(inst.operand_idxs[1])];
  auto *obj = ctx->k->MatMul(lhs, rhs, inst.aux_flags[0], inst.aux_flags[1], nullptr);
  if (!CheckResult(ctx, obj, "MatMul")) return false;
  (*ctx->values)[static_cast<size_t>(inst.result_idx)] = obj;
  return true;
}

bool HandleReduce(const DvmInstruction &inst, BuildCtx *ctx) {
  if (!RequireValidResultIdx(ctx, inst.result_idx, "Invalid result_idx for Reduce instruction")) return false;
  if (!RequireValidValueRefs(ctx, {inst.operand_idxs[0]}, "Invalid operand for Reduce instruction")) return false;
  // Keep ShapeRef object address stable.
  auto &axes_ref = (*ctx->reduceAxesRefs)[static_cast<size_t>(inst.result_idx)];
  axes_ref.data = inst.aux_params.data();
  axes_ref.size = inst.aux_params.size();
  auto *src = (*ctx->values)[static_cast<size_t>(inst.operand_idxs[0])];

  LOG_OUT << "Reduce op: src=" << src << ", axes_size=" << axes_ref.size
          << ", first_axis=" << (axes_ref.size > 0 ? axes_ref.data[0] : -1);

  auto *obj = ctx->k->Reduce(inst.aux_int, src, &axes_ref, inst.aux_flags[0]);
  if (!CheckResult(ctx, obj, "Reduce")) return false;
  (*ctx->values)[static_cast<size_t>(inst.result_idx)] = obj;
  return true;
}

bool HandleCast(const DvmInstruction &inst, BuildCtx *ctx) {
  if (!RequireValidResultIdx(ctx, inst.result_idx, "Invalid result_idx for Cast instruction")) return false;
  if (!RequireValidValueRefs(ctx, {inst.operand_idxs[0]}, "Invalid operand for Cast instruction")) return false;
  auto *src = (*ctx->values)[static_cast<size_t>(inst.operand_idxs[0])];
  auto *obj = ctx->k->Cast(src, static_cast<dvm::DType>(inst.aux_int));
  if (!CheckResult(ctx, obj, "Cast")) return false;
  (*ctx->values)[static_cast<size_t>(inst.result_idx)] = obj;
  return true;
}

bool HandleReshape(const DvmInstruction &inst, BuildCtx *ctx) {
  if (!RequireValidResultIdx(ctx, inst.result_idx, "Invalid result_idx for Reshape instruction")) return false;
  if (!RequireValidValueRefs(ctx, {inst.operand_idxs[0]}, "Invalid operand for Reshape instruction")) return false;
  auto *src = (*ctx->values)[static_cast<size_t>(inst.operand_idxs[0])];
  auto *obj = ctx->k->Reshape(src, const_cast<dvm::ShapeRef *>(ctx->outputShapeRef));
  if (!CheckResult(ctx, obj, "Reshape")) return false;
  (*ctx->values)[static_cast<size_t>(inst.result_idx)] = obj;
  return true;
}

bool HandleBroadcast(const DvmInstruction &inst, BuildCtx *ctx) {
  if (!RequireValidResultIdx(ctx, inst.result_idx, "Invalid result_idx for Broadcast instruction")) return false;
  if (!RequireValidValueRefs(ctx, {inst.operand_idxs[0]}, "Invalid operand for Broadcast instruction")) return false;
  auto *src = (*ctx->values)[static_cast<size_t>(inst.operand_idxs[0])];
  auto *obj = ctx->k->Broadcast(src, const_cast<dvm::ShapeRef *>(ctx->outputShapeRef));
  if (!CheckResult(ctx, obj, "Broadcast")) return false;
  (*ctx->values)[static_cast<size_t>(inst.result_idx)] = obj;
  return true;
}

bool HandleSelect(const DvmInstruction &inst, BuildCtx *ctx) {
  if (!RequireValidResultIdx(ctx, inst.result_idx, "Invalid result_idx for Select instruction")) return false;
  if (!RequireValidValueRefs(ctx, {inst.operand_idxs[0], inst.operand_idxs[1], inst.operand_idxs[2]},
                             "Invalid operands for Select instruction"))
    return false;
  auto *cond = (*ctx->values)[static_cast<size_t>(inst.operand_idxs[0])];
  auto *lhs = (*ctx->values)[static_cast<size_t>(inst.operand_idxs[1])];
  auto *rhs = (*ctx->values)[static_cast<size_t>(inst.operand_idxs[2])];
  auto *obj = ctx->k->Select(cond, lhs, rhs);
  if (!CheckResult(ctx, obj, "Select")) return false;
  (*ctx->values)[static_cast<size_t>(inst.result_idx)] = obj;
  return true;
}

bool HandleUnsupported(const DvmInstruction &inst, BuildCtx *ctx) {
  (void)inst;
  Fail(ctx, "Unsupported OpCode in OpDvmCall build graph");
  return false;
}

const std::array<BuildHandler, kOpCodeCount> kBuildHandlerByOp = {
  // Load operations
  &HandleLoad,         // kLoad
  &HandleUnsupported,  // kSliceLoad
  &HandleUnsupported,  // kStridedSliceLoad
  &HandleUnsupported,  // kMultiLoad

  // Store operations
  &HandleStore,        // kStore
  &HandleUnsupported,  // kPadStore
  &HandleUnsupported,  // kSetStoreInplace

  // Compute operations
  &HandleUnary,        // kUnary
  &HandleBinary,       // kBinary
  &HandleMatMul,       // kMatMul
  &HandleUnsupported,  // kGroupedMatMul
  &HandleReduce,       // kReduce
  &HandleSelect,       // kSelect
  &HandleCast,         // kCast
  &HandleBroadcast,    // kBroadcast
  &HandleReshape,      // kReshape
  &HandleUnsupported,  // kCopy
  &HandleUnsupported,  // kOneHot
  &HandleUnsupported,  // kElemAny

  // Collective communication
  &HandleUnsupported,  // kAllReduce
  &HandleUnsupported,  // kAllGather
  &HandleUnsupported,  // kAllGatherV2
  &HandleUnsupported,  // kReduceScatter

  // Multi-stage operations
  &HandleUnsupported,  // kStageSwitch
  &HandleUnsupported,  // kStageLoad
  &HandleUnsupported,  // kStageStore
  &HandleUnsupported,  // kStagePadStore
};

}  // namespace

void OpDvmCall::Init(const std::vector<const ir::Value *> &inputs, const ir::Value *output) {
  if (inputs.empty()) {
    MRT_THROW("Input list is empty in dvm_call");
  }

  if (!inputs[0]->IsString()) {
    MRT_THROW("First input of dvm_call must be a string payload");
  }

  payload_ = inputs[0]->ToString();
  if (payload_.empty()) {
    MRT_THROW("DVM payload is empty in dvm_call");
  }

  // Store real inputs for future use (skipping the payload string at index 0)
  realInputs_.clear();
  for (size_t i = 1; i < inputs.size(); ++i) {
    realInputs_.push_back(inputs[i]);
  }

  // Parse JSON payload
  DvmKernelPayload parsedPayload = ParseDvmPayload(payload_);
  LOG_OUT << "[OpDvmCall] Parsed payload with " << parsedPayload.instructions.size() << " instructions";

  // Define buildFunc that interprets OpCode and builds DVM kernel graph
  // NOTE: Some DVM ops (e.g., Reduce) take a ShapeRef* for axes/dims and the DVM
  // runtime may keep that pointer beyond build-time. Do NOT pass pointers to
  // stack-allocated ShapeRef storage.
  auto reduceAxesRefs = std::make_shared<std::vector<dvm::ShapeRef>>(parsedPayload.instructions.size());

  auto buildFunc = [parsedPayload, reduceAxesRefs](
                     dvm::Kernel &k, const std::vector<const ir::Value *> &ins, const ir::Value *out,
                     const std::vector<dvm::ShapeRef *> &inputShapeRefs, const dvm::ShapeRef *outputShapeRef,
                     std::vector<dvm::NDObject *> *inputObjs, std::vector<dvm::NDObject *> *outputObjs) {
    (void)ins;
    (void)out;
    std::vector<dvm::NDObject *> values(parsedPayload.instructions.size(), nullptr);
    size_t inputLoadCount = 0;

    for (const auto &inst : parsedPayload.instructions) {
      const auto op_idx = static_cast<size_t>(inst.opcode);
      if (op_idx >= kBuildHandlerByOp.size() || kBuildHandlerByOp[op_idx] == nullptr) {
        MRT_THROW("Unhandled OpCode: ", static_cast<int>(inst.opcode), ", inst.idx=", inst.result_idx);
      }
      BuildCtx ctx;
      ctx.k = &k;
      ctx.inputShapeRefs = &inputShapeRefs;
      ctx.outputShapeRef = outputShapeRef;
      ctx.values = &values;
      ctx.inputLoadCount = &inputLoadCount;
      ctx.inputObjs = inputObjs;
      ctx.outputObjs = outputObjs;
      ctx.reduceAxesRefs = reduceAxesRefs.get();
      if (!kBuildHandlerByOp[op_idx](inst, &ctx)) {
        break;
      }
    }

    LOG_OUT << "[OpDvmCall] Built kernel graph with " << inputObjs->size() << " inputs, " << outputObjs->size()
            << " outputs";
  };

  // Decide kernel type.
  //
  // IMPORTANT: MindSpore uses kStaticMix/kDynMix when the graph contains MatMul.
  // If we keep kStaticShape/kDynShape for MatMul, DVM may fail to infer MatMul
  // output shape (Dump shows "[]") and then segfault in Store CodeGen.
  auto kernelType = KernelTypeFromString(parsedPayload.kernel_type);
  bool hasMatMul =
    std::any_of(parsedPayload.instructions.begin(), parsedPayload.instructions.end(),
                [](const auto &inst) { return inst.opcode == kMatMul || inst.opcode == kGroupedMatMul; });
  if (hasMatMul) {
    if (kernelType == dvm::kStaticShape) {
      LOG_OUT << "[OpDvmCall] kernel_type='" << parsedPayload.kernel_type
              << "' but MatMul found; upgrading to static_mix";
      kernelType = dvm::kStaticMix;
    } else if (kernelType == dvm::kDynShape) {
      LOG_OUT << "[OpDvmCall] kernel_type='" << parsedPayload.kernel_type << "' but MatMul found; upgrading to dyn_mix";
      kernelType = dvm::kDynMix;
    }
  }
  LOG_OUT << "[OpDvmCall] Using DVM kernel type enum=" << static_cast<int>(kernelType);

  dvmOp_ = std::make_unique<DvmOp>(kernelType, buildFunc);
  dvmOp_->Init(realInputs_, output);

  LOG_OUT << "[OpDvmCall] Initialized DvmOp with parsed payload";
}

OpsErrorCode OpDvmCall::InferShape(const std::vector<const ir::Value *> &input, ir::Value *output) {
  (void)input;
  if (dvmOp_) {
    return dvmOp_->InferShape(realInputs_, output);
  }
  return OpsErrorCode::SUCCESS;
}

OpsErrorCode OpDvmCall::CalcWorkspace(const std::vector<const ir::Value *> &input, const ir::Value *output,
                                      size_t *workspaceSize) {
  (void)input;
  if (dvmOp_) {
    return dvmOp_->CalcWorkspace(realInputs_, output, workspaceSize);
  }
  *workspaceSize = 0;
  return OpsErrorCode::SUCCESS;
}

OpsErrorCode OpDvmCall::Launch(const std::vector<const ir::Value *> &input, void *workspace, size_t workspaceSize,
                               ir::Value *output, void *stream) {
  (void)input;
  if (dvmOp_) {
    return dvmOp_->Launch(realInputs_, workspace, workspaceSize, output, stream);
  }
  MRT_THROW("OpDvmCall::Launch: dvmOp_ is not initialized");
  return OpsErrorCode::UNKNOWN_ERROR;
}

}  // namespace ops
}  // namespace mrt
