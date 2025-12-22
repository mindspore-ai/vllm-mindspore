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

#include "ops/ascend/dvm/dvm_payload.h"

#include <algorithm>
#include <array>
#include <string>
#include <string_view>
#include <unordered_map>

#include "nlohmann/json.hpp"
#include "common/throw.h"
#include "ops/ascend/dvm/prebuild/dvm.h"

using json = nlohmann::json;

namespace mrt {
namespace ops {

namespace {

constexpr size_t kOpCodeCount = static_cast<size_t>(DvmOpCode::kOpCodeCount);

template <typename T>
struct MappingEntry {
  std::string_view name;
  T value;
};

const std::unordered_map<std::string_view, DvmOpCode> kOpCodeByName = {
  // Load operations
  {"load", kLoad},
  {"slice_load", kSliceLoad},
  {"strided_slice_load", kStridedSliceLoad},
  {"multi_load", kMultiLoad},

  // Store operations
  {"store", kStore},
  {"pad_store", kPadStore},
  {"set_store_inplace", kSetStoreInplace},

  // Compute operations
  {"unary", kUnary},
  {"binary", kBinary},
  {"matmul", kMatMul},
  {"grouped_matmul", kGroupedMatMul},
  {"reduce", kReduce},
  {"select", kSelect},
  {"cast", kCast},
  {"broadcast", kBroadcast},
  {"reshape", kReshape},
  {"copy", kCopy},
  {"onehot", kOneHot},
  {"elem_any", kElemAny},

  // Collective communication
  {"allreduce", kAllReduce},
  {"allgather", kAllGather},
  {"allgather_v2", kAllGatherV2},
  {"reduce_scatter", kReduceScatter},

  // Multi-stage operations
  {"stage_switch", kStageSwitch},
  {"stage_load", kStageLoad},
  {"stage_store", kStageStore},
  {"stage_pad_store", kStagePadStore},
};

const std::array<std::string_view, kOpCodeCount> kOpNameByCode = {
  // Load operations
  "load",                // kLoad
  "slice_load",          // kSliceLoad
  "strided_slice_load",  // kStridedSliceLoad
  "multi_load",          // kMultiLoad

  // Store operations
  "store",              // kStore
  "pad_store",          // kPadStore
  "set_store_inplace",  // kSetStoreInplace

  // Compute operations
  "unary",           // kUnary
  "binary",          // kBinary
  "matmul",          // kMatMul
  "grouped_matmul",  // kGroupedMatMul
  "reduce",          // kReduce
  "select",          // kSelect
  "cast",            // kCast
  "broadcast",       // kBroadcast
  "reshape",         // kReshape
  "copy",            // kCopy
  "onehot",          // kOneHot
  "elem_any",        // kElemAny

  // Collective communication
  "allreduce",       // kAllReduce
  "allgather",       // kAllGather
  "allgather_v2",    // kAllGatherV2
  "reduce_scatter",  // kReduceScatter

  // Multi-stage operations
  "stage_switch",     // kStageSwitch
  "stage_load",       // kStageLoad
  "stage_store",      // kStageStore
  "stage_pad_store",  // kStagePadStore
};

constexpr std::array<MappingEntry<int32_t>, 6> kDTypes = {{
  {"float16", dvm::kFloat16},
  {"bfloat16", dvm::kBFloat16},
  {"float32", dvm::kFloat32},
  {"int32", dvm::kInt32},
  {"int64", dvm::kInt64},
  {"bool", dvm::kBool},
}};

constexpr std::array<MappingEntry<int32_t>, 15> kBinaryOps = {{
  {"equal", dvm::kEqual},
  {"not_equal", dvm::kNotEqual},
  {"greater", dvm::kGreater},
  {"greater_equal", dvm::kGreaterEqual},
  {"less", dvm::kLess},
  {"less_equal", dvm::kLessEqual},
  {"add", dvm::kAdd},
  {"sub", dvm::kSub},
  {"mul", dvm::kMul},
  {"div", dvm::kDiv},
  {"pow", dvm::kPow},
  {"maximum", dvm::kMaximum},
  {"minimum", dvm::kMinimum},
  {"logical_and", dvm::kLogicalAnd},
  {"logical_or", dvm::kLogicalOr},
}};

constexpr std::array<MappingEntry<int32_t>, 11> kUnaryOps = {{
  {"sqrt", dvm::kSqrt},
  {"abs", dvm::kAbs},
  {"log", dvm::kLog},
  {"exp", dvm::kExp},
  {"reciprocal", dvm::kReciprocal},
  {"is_finite", dvm::kIsFinite},
  {"logical_not", dvm::kLogicalNot},
  {"round", dvm::kRound},
  {"floor", dvm::kFloor},
  {"ceil", dvm::kCeil},
  {"trunc", dvm::kTrunc},
}};

constexpr std::array<MappingEntry<int32_t>, 1> kReduceOps = {{
  {"sum", dvm::kSum},
}};

template <typename T, size_t N>
T LookupByName(const std::array<MappingEntry<T>, N> &table, std::string_view name, const char *what) {
  auto it = std::find_if(table.begin(), table.end(), [name](const auto &entry) { return entry.name == name; });
  if (it != table.end()) {
    return it->value;
  }
  MRT_THROW("Unknown ", what, ": ", name);
}

template <typename T, size_t N>
std::string_view LookupByValue(const std::array<MappingEntry<T>, N> &table, T value, std::string_view default_value) {
  auto it = std::find_if(table.begin(), table.end(), [value](const auto &entry) { return entry.value == value; });
  if (it != table.end()) {
    return it->name;
  }
  return default_value;
}

DvmOpCode StringToOpCode(const std::string &op) {
  auto it = kOpCodeByName.find(std::string_view{op});
  if (it == kOpCodeByName.end()) {
    MRT_THROW("Unknown OpCode: ", op);
  }
  return it->second;
}

int32_t StringToDType(const std::string &dtype) { return LookupByName(kDTypes, dtype, "DType"); }

int32_t StringToBinaryOp(const std::string &op) { return LookupByName(kBinaryOps, op, "Binary Op"); }

int32_t StringToUnaryOp(const std::string &op) { return LookupByName(kUnaryOps, op, "Unary Op"); }

int32_t StringToReduceOp(const std::string &op) { return LookupByName(kReduceOps, op, "Reduce Op"); }

using ParseInstFn = void (*)(const json &, DvmInstruction *);

const json *GetAttrsPtr(const json &inst_json) {
  if (!inst_json.contains("attrs")) return nullptr;
  const auto &attrs = inst_json["attrs"];
  if (attrs.is_null()) return nullptr;
  if (!attrs.is_object()) {
    MRT_THROW("Instruction 'attrs' must be an object");
  }
  return &attrs;
}

void ParseInputsField(const json &inst_json, DvmInstruction *inst) {
  if (!inst_json.contains("inputs")) return;
  const auto &inputs = inst_json["inputs"];
  if (!inputs.is_array()) {
    MRT_THROW("Instruction 'inputs' must be an array");
  }
  if (inputs.size() > kMaxOperands) {
    MRT_THROW("Instruction 'inputs' length must be <= ", kMaxOperands);
  }
  for (size_t i = 0; i < inputs.size(); ++i) {
    inst->operand_idxs[i] = inputs[i].get<int32_t>();
  }
}

void ParseLoad(const json &inst_json, DvmInstruction *inst) {
  const auto *attrs = GetAttrsPtr(inst_json);
  if (attrs != nullptr && attrs->contains("dtype")) {
    inst->aux_int = StringToDType((*attrs)["dtype"].get<std::string>());
    return;
  }
  // Backward compatible: old schema used top-level "dtype".
  if (inst_json.contains("dtype")) {
    inst->aux_int = StringToDType(inst_json["dtype"].get<std::string>());
    return;
  }
  MRT_THROW("Load instruction missing attrs.dtype");
}

void ParseStore(const json &inst_json, DvmInstruction *inst) {
  // New schema: operand_idxs[0] is filled by "inputs":[src]
  if (inst->operand_idxs[0] != -1) return;
  // Backward compatible: old schema used top-level "src".
  if (inst_json.contains("src")) {
    inst->operand_idxs[0] = inst_json["src"].get<int32_t>();
  }
}

void ParseBinary(const json &inst_json, DvmInstruction *inst) {
  const auto *attrs = GetAttrsPtr(inst_json);
  if (attrs != nullptr && attrs->contains("type")) {
    inst->aux_int = StringToBinaryOp((*attrs)["type"].get<std::string>());
  } else if (inst_json.contains("type")) {
    // Backward compatible: old schema used top-level "type".
    inst->aux_int = StringToBinaryOp(inst_json["type"].get<std::string>());
  }
  // Backward compatible: old schema used "lhs"/"rhs".
  if (!inst_json.contains("inputs")) {
    if (inst_json.contains("lhs")) {
      inst->operand_idxs[0] = inst_json["lhs"].get<int32_t>();
    }
    if (inst_json.contains("rhs")) {
      inst->operand_idxs[1] = inst_json["rhs"].get<int32_t>();
    }
  }
}

void ParseUnary(const json &inst_json, DvmInstruction *inst) {
  const auto *attrs = GetAttrsPtr(inst_json);
  if (attrs != nullptr && attrs->contains("type")) {
    inst->aux_int = StringToUnaryOp((*attrs)["type"].get<std::string>());
  } else if (inst_json.contains("type")) {
    // Backward compatible: old schema used top-level "type".
    inst->aux_int = StringToUnaryOp(inst_json["type"].get<std::string>());
  }
  // Backward compatible: old schema used top-level "src".
  if (!inst_json.contains("inputs") && inst_json.contains("src")) {
    inst->operand_idxs[0] = inst_json["src"].get<int32_t>();
  }
}

void ParseMatMul(const json &inst_json, DvmInstruction *inst) {
  // Backward compatible: old schema used "lhs"/"rhs".
  if (!inst_json.contains("inputs")) {
    if (inst_json.contains("lhs")) {
      inst->operand_idxs[0] = inst_json["lhs"].get<int32_t>();
    }
    if (inst_json.contains("rhs")) {
      inst->operand_idxs[1] = inst_json["rhs"].get<int32_t>();
    }
  }
  const auto *attrs = GetAttrsPtr(inst_json);
  if (attrs != nullptr) {
    if (attrs->contains("trans_a")) {
      inst->aux_flags[0] = (*attrs)["trans_a"].get<bool>();
    }
    if (attrs->contains("trans_b")) {
      inst->aux_flags[1] = (*attrs)["trans_b"].get<bool>();
    }
  } else {
    // Backward compatible: old schema used top-level "trans_a"/"trans_b".
    if (inst_json.contains("trans_a")) {
      inst->aux_flags[0] = inst_json["trans_a"].get<bool>();
    }
    if (inst_json.contains("trans_b")) {
      inst->aux_flags[1] = inst_json["trans_b"].get<bool>();
    }
  }
}

void ParseReduce(const json &inst_json, DvmInstruction *inst) {
  const auto *attrs = GetAttrsPtr(inst_json);
  if (attrs != nullptr) {
    if (attrs->contains("type")) {
      inst->aux_int = StringToReduceOp((*attrs)["type"].get<std::string>());
    }
    if (attrs->contains("axes")) {
      inst->aux_params = (*attrs)["axes"].get<std::vector<int64_t>>();
    }
    if (attrs->contains("keepdims")) {
      inst->aux_flags[0] = (*attrs)["keepdims"].get<bool>();
    }
  } else {
    // Backward compatible: old schema used top-level keys.
    if (inst_json.contains("type")) {
      inst->aux_int = StringToReduceOp(inst_json["type"].get<std::string>());
    }
    if (inst_json.contains("axes")) {
      inst->aux_params = inst_json["axes"].get<std::vector<int64_t>>();
    }
    if (inst_json.contains("keepdims")) {
      inst->aux_flags[0] = inst_json["keepdims"].get<bool>();
    }
  }
  // Backward compatible: old schema used top-level "src".
  if (!inst_json.contains("inputs") && inst_json.contains("src")) {
    inst->operand_idxs[0] = inst_json["src"].get<int32_t>();
  }
}

void ParseCast(const json &inst_json, DvmInstruction *inst) {
  // Backward compatible: old schema used top-level "src".
  if (!inst_json.contains("inputs") && inst_json.contains("src")) {
    inst->operand_idxs[0] = inst_json["src"].get<int32_t>();
  }
  const auto *attrs = GetAttrsPtr(inst_json);
  if (attrs != nullptr && attrs->contains("dtype")) {
    inst->aux_int = StringToDType((*attrs)["dtype"].get<std::string>());
    return;
  }
  // Backward compatible: old schema used top-level "dtype".
  if (inst_json.contains("dtype")) {
    inst->aux_int = StringToDType(inst_json["dtype"].get<std::string>());
    return;
  }
  MRT_THROW("Cast instruction missing attrs.dtype");
}

void ParseReshapeOrBroadcast(const json &inst_json, DvmInstruction *inst) {
  // New schema uses "inputs":[src] (parsed in ParseInputsField).
  // Backward compatible: old schema used top-level "src".
  if (!inst_json.contains("inputs") && inst_json.contains("src")) {
    inst->operand_idxs[0] = inst_json["src"].get<int32_t>();
  }
  // Shape is provided at runtime via ShapeRef.
}

void ParseSelect(const json &inst_json, DvmInstruction *inst) {
  // New schema: inputs=[cond, lhs, rhs].
  // Backward compatible: old schema used cond/lhs/rhs.
  if (!inst_json.contains("inputs")) {
    if (inst_json.contains("cond")) {
      inst->operand_idxs[0] = inst_json["cond"].get<int32_t>();
    }
    if (inst_json.contains("lhs")) {
      inst->operand_idxs[1] = inst_json["lhs"].get<int32_t>();
    }
    if (inst_json.contains("rhs")) {
      inst->operand_idxs[2] = inst_json["rhs"].get<int32_t>();
    }
  }
}

void ParseUnsupported(const json &inst_json, DvmInstruction *inst) {
  (void)inst_json;
  (void)inst;
}

const std::array<ParseInstFn, kOpCodeCount> kParseInstByOp = {
  // Load operations
  &ParseLoad,         // kLoad
  &ParseUnsupported,  // kSliceLoad
  &ParseUnsupported,  // kStridedSliceLoad
  &ParseUnsupported,  // kMultiLoad

  // Store operations
  &ParseStore,        // kStore
  &ParseUnsupported,  // kPadStore
  &ParseUnsupported,  // kSetStoreInplace

  // Compute operations
  &ParseUnary,               // kUnary
  &ParseBinary,              // kBinary
  &ParseMatMul,              // kMatMul
  &ParseUnsupported,         // kGroupedMatMul
  &ParseReduce,              // kReduce
  &ParseSelect,              // kSelect
  &ParseCast,                // kCast
  &ParseReshapeOrBroadcast,  // kBroadcast
  &ParseReshapeOrBroadcast,  // kReshape
  &ParseUnsupported,         // kCopy
  &ParseUnsupported,         // kOneHot
  &ParseUnsupported,         // kElemAny

  // Collective communication
  &ParseUnsupported,  // kAllReduce
  &ParseUnsupported,  // kAllGather
  &ParseUnsupported,  // kAllGatherV2
  &ParseUnsupported,  // kReduceScatter

  // Multi-stage operations
  &ParseUnsupported,  // kStageSwitch
  &ParseUnsupported,  // kStageLoad
  &ParseUnsupported,  // kStageStore
  &ParseUnsupported,  // kStagePadStore
};

using SerializeInstFn = void (*)(const DvmInstruction &, json *);

void SerializeLoad(const DvmInstruction &inst, json *inst_json) {
  (*inst_json)["op"] = "load";
  (*inst_json)["inputs"] = json::array();
  (*inst_json)["attrs"] = json::object({{"dtype", std::string(LookupByValue(kDTypes, inst.aux_int, "float32"))}});
}

void SerializeStore(const DvmInstruction &inst, json *inst_json) {
  (*inst_json)["op"] = "store";
  (*inst_json)["inputs"] = json::array({inst.operand_idxs[0]});
  (*inst_json)["attrs"] = json::object();
}

void SerializeBinary(const DvmInstruction &inst, json *inst_json) {
  (*inst_json)["op"] = "binary";
  (*inst_json)["inputs"] = json::array({inst.operand_idxs[0], inst.operand_idxs[1]});
  (*inst_json)["attrs"] = json::object({{"type", std::string(LookupByValue(kBinaryOps, inst.aux_int, "add"))}});
}

void SerializeUnary(const DvmInstruction &inst, json *inst_json) {
  (*inst_json)["op"] = "unary";
  (*inst_json)["inputs"] = json::array({inst.operand_idxs[0]});
  (*inst_json)["attrs"] = json::object({{"type", std::string(LookupByValue(kUnaryOps, inst.aux_int, "sqrt"))}});
}

void SerializeMatMul(const DvmInstruction &inst, json *inst_json) {
  (*inst_json)["op"] = "matmul";
  (*inst_json)["inputs"] = json::array({inst.operand_idxs[0], inst.operand_idxs[1]});
  (*inst_json)["attrs"] = json::object({{"trans_a", inst.aux_flags[0]}, {"trans_b", inst.aux_flags[1]}});
}

void SerializeReduce(const DvmInstruction &inst, json *inst_json) {
  (*inst_json)["op"] = "reduce";
  (*inst_json)["inputs"] = json::array({inst.operand_idxs[0]});
  (*inst_json)["attrs"] = json::object({
    {"type", std::string(LookupByValue(kReduceOps, inst.aux_int, "sum"))},
    {"axes", inst.aux_params},
    {"keepdims", inst.aux_flags[0]},
  });
}

void SerializeCast(const DvmInstruction &inst, json *inst_json) {
  (*inst_json)["op"] = "cast";
  (*inst_json)["inputs"] = json::array({inst.operand_idxs[0]});
  (*inst_json)["attrs"] = json::object({{"dtype", std::string(LookupByValue(kDTypes, inst.aux_int, "float32"))}});
}

void SerializeReshape(const DvmInstruction &inst, json *inst_json) {
  (*inst_json)["op"] = "reshape";
  (*inst_json)["inputs"] = json::array({inst.operand_idxs[0]});
  (*inst_json)["attrs"] = json::object();
}

void SerializeBroadcast(const DvmInstruction &inst, json *inst_json) {
  (*inst_json)["op"] = "broadcast";
  (*inst_json)["inputs"] = json::array({inst.operand_idxs[0]});
  (*inst_json)["attrs"] = json::object();
}

void SerializeSelect(const DvmInstruction &inst, json *inst_json) {
  (*inst_json)["op"] = "select";
  (*inst_json)["inputs"] = json::array({inst.operand_idxs[0], inst.operand_idxs[1], inst.operand_idxs[2]});
  (*inst_json)["attrs"] = json::object();
}

void SerializeUnsupported(const DvmInstruction &inst, json *inst_json) {
  (void)inst;
  (*inst_json)["op"] = "unsupported";
  (*inst_json)["inputs"] = json::array();
  (*inst_json)["attrs"] = json::object();
}

const std::array<SerializeInstFn, kOpCodeCount> kSerializeInstByOp = {
  // Load operations
  &SerializeLoad,         // kLoad
  &SerializeUnsupported,  // kSliceLoad
  &SerializeUnsupported,  // kStridedSliceLoad
  &SerializeUnsupported,  // kMultiLoad

  // Store operations
  &SerializeStore,        // kStore
  &SerializeUnsupported,  // kPadStore
  &SerializeUnsupported,  // kSetStoreInplace

  // Compute operations
  &SerializeUnary,        // kUnary
  &SerializeBinary,       // kBinary
  &SerializeMatMul,       // kMatMul
  &SerializeUnsupported,  // kGroupedMatMul
  &SerializeReduce,       // kReduce
  &SerializeSelect,       // kSelect
  &SerializeCast,         // kCast
  &SerializeBroadcast,    // kBroadcast
  &SerializeReshape,      // kReshape
  &SerializeUnsupported,  // kCopy
  &SerializeUnsupported,  // kOneHot
  &SerializeUnsupported,  // kElemAny

  // Collective communication
  &SerializeUnsupported,  // kAllReduce
  &SerializeUnsupported,  // kAllGather
  &SerializeUnsupported,  // kAllGatherV2
  &SerializeUnsupported,  // kReduceScatter

  // Multi-stage operations
  &SerializeUnsupported,  // kStageSwitch
  &SerializeUnsupported,  // kStageLoad
  &SerializeUnsupported,  // kStageStore
  &SerializeUnsupported,  // kStagePadStore
};

}  // namespace

DvmKernelPayload ParseDvmPayload(const std::string &json_str) {
  DvmKernelPayload payload;

  // Non-throwing parse to avoid exception-based control-flow.
  auto j = json::parse(json_str, nullptr, false);
  if (j.is_discarded()) {
    MRT_THROW("Failed to parse DVM payload JSON");
  }

  // Parse version and kernel_type
  if (j.contains("version")) {
    payload.version = j["version"].get<int32_t>();
  }

  if (j.contains("kernel_type")) {
    payload.kernel_type = j["kernel_type"].get<std::string>();
  }

  // Parse instructions
  if (!j.contains("instructions")) {
    MRT_THROW("Missing 'instructions' field in DVM payload");
  }

  for (auto &inst_json : j["instructions"]) {
    DvmInstruction inst;

    if (!inst_json.contains("op") || !inst_json.contains("idx")) {
      MRT_THROW("Instruction missing 'op' or 'idx' field");
    }

    std::string op = inst_json["op"].get<std::string>();
    inst.opcode = StringToOpCode(op);
    inst.result_idx = inst_json["idx"].get<int32_t>();

    // New schema: parse common operands from "inputs" first.
    ParseInputsField(inst_json, &inst);

    // Parse opcode-specific fields
    const auto op_idx = static_cast<size_t>(inst.opcode);
    if (op_idx >= kParseInstByOp.size() || kParseInstByOp[op_idx] == nullptr) {
      LOG_EXCEPTION << "Unhandled OpCode: " << static_cast<int>(inst.opcode);
    }
    kParseInstByOp[op_idx](inst_json, &inst);

    payload.instructions.push_back(inst);
  }

  // Parse input/output indices
  if (j.contains("input_indices")) {
    payload.input_indices = j["input_indices"].get<std::vector<int32_t>>();
  }

  if (j.contains("output_indices")) {
    payload.output_indices = j["output_indices"].get<std::vector<int32_t>>();
  }

  LOG_OUT << "Parsed DVM payload: " << payload.instructions.size() << " instructions, " << payload.input_indices.size()
          << " inputs, " << payload.output_indices.size() << " outputs";

  return payload;
}

std::string SerializeDvmPayload(const DvmKernelPayload &payload) {
  json j;
  j["version"] = payload.version;
  j["kernel_type"] = payload.kernel_type;

  std::vector<json> instructions;
  for (const auto &inst : payload.instructions) {
    json inst_json;
    inst_json["idx"] = inst.result_idx;
    // New schema requires these two fields for all instructions.
    inst_json["inputs"] = json::array();
    inst_json["attrs"] = json::object();

    // Basic fields based on opcode
    const auto op_idx = static_cast<size_t>(inst.opcode);
    if (op_idx < kSerializeInstByOp.size() && kSerializeInstByOp[op_idx] != nullptr) {
      kSerializeInstByOp[op_idx](inst, &inst_json);
    } else if (op_idx < kOpNameByCode.size()) {
      inst_json["op"] = std::string(kOpNameByCode[op_idx]);
    } else {
      inst_json["op"] = "unknown";
    }

    instructions.push_back(inst_json);
  }

  j["instructions"] = instructions;
  j["input_indices"] = payload.input_indices;
  j["output_indices"] = payload.output_indices;

  return j.dump(2);  // Pretty print with 2-space indentation
}
}  // namespace ops
}  // namespace mrt
