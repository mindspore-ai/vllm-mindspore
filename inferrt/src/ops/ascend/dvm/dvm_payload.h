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

#ifndef __OPS_ASCEND_DVM_DVM_PAYLOAD_H__
#define __OPS_ASCEND_DVM_DVM_PAYLOAD_H__

#include <cstdint>
#include <string>
#include <vector>

namespace mrt {
namespace ops {

// OpCode types matching dvm::Kernel API
enum DvmOpCode {
  // Load operations
  kLoad = 0,
  kSliceLoad,
  kStridedSliceLoad,
  kMultiLoad,

  // Store operations
  kStore,
  kPadStore,
  kSetStoreInplace,

  // Compute operations
  kUnary,
  kBinary,
  kMatMul,
  kGroupedMatMul,
  kReduce,
  kSelect,
  kCast,
  kBroadcast,
  kReshape,
  kCopy,
  kOneHot,
  kElemAny,

  // Collective communication
  kAllReduce,
  kAllGather,
  kAllGatherV2,
  kReduceScatter,

  // Multi-stage operations
  kStageSwitch,
  kStageLoad,
  kStageStore,
  kStagePadStore,

  // Count sentinel
  kOpCodeCount,
};

// Maximum number of operands and auxiliary flags per instruction
constexpr size_t kMaxOperands = 4;
constexpr size_t kMaxAuxFlags = 4;

// Represents a single DVM instruction.
//
// JSON schema (recommended):
//   {
//     "op": "load" | "binary" | "matmul" | "store" | ...,
//     "idx": <int>,                 // result slot (value table index)
//     "inputs": [<int>, ...],       // operand idx list (maps to operand_idxs[])
//     "attrs": { ... }              // op attributes (dtype/type/trans_a/axes/...)
//   }
//
// Backward-compatibility: parser may also accept legacy fields like lhs/rhs/src/dtype.
struct DvmInstruction {
  DvmOpCode opcode;
  int32_t result_idx;                  // Position in value table
  int32_t operand_idxs[kMaxOperands];  // Operand references (-1 means unused)
  int32_t aux_int;                     // Auxiliary parameter (opType/dtype/axis etc)
  bool aux_flags[kMaxAuxFlags];        // Boolean parameters (trans_a, trans_b, keepdims etc)
  std::vector<int64_t> aux_params;     // Variable-length params (reduce axes, transpose perms etc)
  // Optional: target shape reference for shape-driven ops like reshape/broadcast.
  //
  // JSON schema:
  //   "attrs": {
  //     "shape_ref": {
  //       "output_pos": <int>   // use output tuple position as target shape
  //       OR
  //       "input_pos":  <int>   // use input tensor position as target shape
  //       OR
  //       "dims": [<int>, ...]  // constant target shape
  //     }
  //   }
  //
  // Exactly one of (shape_ref_output_pos, shape_ref_input_pos, shape_ref_dims) should be specified.
  int32_t shape_ref_output_pos;
  int32_t shape_ref_input_pos;
  std::vector<int64_t> shape_ref_dims;

  DvmInstruction()
      : opcode(kLoad), result_idx(-1), aux_int(0), shape_ref_output_pos(-1), shape_ref_input_pos(-1), shape_ref_dims() {
    for (size_t i = 0; i < kMaxOperands; ++i) {
      operand_idxs[i] = -1;
    }
    for (size_t i = 0; i < kMaxAuxFlags; ++i) {
      aux_flags[i] = false;
    }
  }
};

// Complete kernel payload
struct DvmKernelPayload {
  int32_t version;
  std::string kernel_type;
  std::vector<DvmInstruction> instructions;
  std::vector<int32_t> input_indices;   // Load instruction indices
  std::vector<int32_t> output_indices;  // Store instruction indices

  DvmKernelPayload() : version(1), kernel_type("static_shape") {}
};

// Parse JSON payload into structured format
DvmKernelPayload ParseDvmPayload(const std::string &json_str);

// Serialize payload to JSON (for debugging)
std::string SerializeDvmPayload(const DvmKernelPayload &payload);

}  // namespace ops
}  // namespace mrt

#endif  // __OPS_ASCEND_DVM_DVM_PAYLOAD_H__
