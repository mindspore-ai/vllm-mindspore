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

#ifndef __OPS_CPU_CPU_CUSTOM_OPERATOR_H__
#define __OPS_CPU_CPU_CUSTOM_OPERATOR_H__

#include <string>
#include <vector>

#include "ops/operator.h"

namespace mrt {
namespace ops {

/**
 * @brief CPU Custom operator base class.
 * Users need to inherit this class and implement Launch methods,
 * Note: InferShape function is not concerned, shape inference is handled automatically by the system.
 */
class DA_API CPUCustomOperator : public Operator {
 public:
  CPUCustomOperator() = default;
  ~CPUCustomOperator() override = default;

  /**
   * @brief Launch the custom operator kernel on CPU.
   *
   * @param input Vector of input ir::Value pointers.
   * @param workspace Pointer to the workspace memory.
   * @param workspaceSize Size of the workspace memory in bytes.
   * @param output Pointer to the output ir::Value.
   * @param stream Pointer to the stream for asynchronous execution.
   */
  virtual OpsErrorCode Launch(const std::vector<const ir::Value *> &input, void *workspace, size_t workspaceSize,
                              ir::Value *output, void *stream) = 0;
};
}  // namespace ops
}  // namespace mrt

#endif  // __OPS_CPU_CPU_CUSTOM_OPERATOR_H__
