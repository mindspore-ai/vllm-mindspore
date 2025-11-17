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

#ifndef __OPS_ASCEND_LOWERED_AUTO_LOWERED_OP_H__
#define __OPS_ASCEND_LOWERED_AUTO_LOWERED_OP_H__

#include <string>
#include <memory>

#include "common/visible.h"
#include "ops/operator.h"
#include "ops/op_register.h"
#include "ops/ascend/lowered/lowered_kernel_executor.h"

namespace mrt::ops {
class AutoLoweredOp : public Operator {
 public:
  /**
   * @brief Construct with kernel specification ID
   * @param specId Kernel spec ID registered in KernelRegistry
   */
  explicit AutoLoweredOp(const std::string &specId);

  ~AutoLoweredOp() override = default;

  /**
   * @brief Calculate workspace size
   *
   * Delegates to LoweredKernelExecutor, which:
   * - For dynamic shape: loads kernel, computes tiling, returns tiling data size
   * - For static shape: returns 0
   *
   * @param input Input values
   * @param output Output value
   * @param workspaceSize Output: workspace size in bytes
   * @return SUCCESS on success, error code otherwise
   */
  OpsErrorCode CalcWorkspace(const std::vector<const ir::Value *> &input, const ir::Value *output,
                             size_t *workspaceSize) override;

  /**
   * @brief Launch the kernel on device
   *
   * Delegates to LoweredKernelExecutor, which:
   * 1. Builds kernel arguments (memref structures + tiling args)
   * 2. Calls Host API function with proper blockDim
   *
   * @param input Input values
   * @param workspace Workspace buffer
   * @param workspaceSize Workspace buffer size
   * @param output Output value
   * @param stream Execution stream
   * @return SUCCESS on success, error code otherwise
   */
  OpsErrorCode Launch(const std::vector<const ir::Value *> &input, void *workspace, size_t workspaceSize,
                      ir::Value *output, void *stream) override;

 private:
  std::string specId_;                              // Kernel specification ID
  std::unique_ptr<LoweredKernelExecutor> executor_;  // Executor for this kernel
};
}  // namespace mrt::ops

#endif  // __OPS_ASCEND_LOWERED_AUTO_LOWERED_OP_H__
