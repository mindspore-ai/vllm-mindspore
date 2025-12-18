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

#ifndef __OPS_ASCEND_DVM_OP_DVM_CALL_H__
#define __OPS_ASCEND_DVM_OP_DVM_CALL_H__

#include <memory>
#include <string>
#include <vector>

#include "ops/op_register.h"

namespace mrt {
namespace ops {

/**
 * @brief OpDvmCall - Standard IR operator for invoking DVM fused kernels
 *
 * Morphology aligned with OpLinalgCall:
 * - inputs[0]: DVM OpCode payload (JSON string)
 * - inputs[1...N]: Actual input tensors
 */
class OpDvmCall : public Operator {
 public:
  OpDvmCall() = default;
  ~OpDvmCall() override = default;

  /**
   * @brief Initialize dvm_call operator
   * @param inputs Input list:
   *   - inputs[0]: DVM Payload (String type Value)
   *   - inputs[1...N]: Actual input tensors
   * @param output Output tensor
   */
  void Init(const std::vector<const ir::Value *> &inputs, const ir::Value *output);

  OpsErrorCode InferShape(const std::vector<const ir::Value *> &input, ir::Value *output) override;

  OpsErrorCode CalcWorkspace(const std::vector<const ir::Value *> &input, const ir::Value *output,
                             size_t *workspaceSize);

  OpsErrorCode Launch(const std::vector<const ir::Value *> &input, void *workspace, size_t workspaceSize,
                      ir::Value *output, void *stream);

 protected:
  std::string payload_;                        // Stored DVM payload (JSON)
  std::unique_ptr<Operator> dvmOp_;            // DvmOp instance
  std::vector<const ir::Value *> realInputs_;  // Actual inputs (excluding payload parameter)
};

}  // namespace ops
}  // namespace mrt

#endif  // __OPS_ASCEND_DVM_OP_DVM_CALL_H__
