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

#ifndef __KERNEL_KERNEL_H__
#define __KERNEL_KERNEL_H__

#include <functional>
#include <string>
#include <vector>

#include "ops/op_def/ops_name.h"
#include "common/visible.h"
#include "ir/graph.h"

namespace mrt {
namespace ops {

// Deprecated Interface for kernel
class DAKernel {
 public:
  explicit DAKernel(ir::NodePtr node) : node_(node) {}
  virtual ~DAKernel() = default;

  virtual void Init() = 0;
  virtual void InferShape() = 0;
  virtual void Resize() = 0;
  virtual void Launch() = 0;

 protected:
  ir::NodePtr node_;
};

// Operator-related error codes. The error types within them will be further expanded in the future.
enum OpsErrorCode {
  SUCCESS = 0,
  INVALID_PARAM,
  INVALID_SHAPE,
  INVALID_DEVICE_ADDR,
  UNKNOWN_ERROR = 1000

};

// Need to be deleted in the future.
using OpName = Op;

// @brief Abstract base class representing a computational kernel. A Operator encapsulates the core computation logic
// for a specific operator. Derived classes must implement shape inference and launch operations. Kernels of different
// device types share the InferShape function, but need to implement their respective Launch functions.
class Operator {
 public:
  Operator(const OpName &op) : op_(op) {}
  virtual ~Operator() = default;

  /**
   *  @brief Infer the output shape based on input shape or value.
   *  @param input Vector of pointers to input data.
   *  @param output Pointer to the output data, the inferred output shape needs to be updated to the output. Note: The
   *  output may be one of types such as Tensor, Tuple, etc.
   *  @return OpsErrorCode Error code indicating success or failure of shape inference.
   */
  virtual OpsErrorCode InferShape(const std::vector<ir::Value> &input, ir::Value output) = 0;

  /**
   * @brief Calculate total workspace memory size requirements for the kernel computation.
   * @param input Vector of pointers to input data.
   * @param output Pointer to the output data.
   * @param workspace_size Pointer to the workspace memory size, the workspace memory size in bytes needs to be updated
   * to the variable pointed by `workspace_size`.
   * @return OpsErrorCode Error code indicating success or failure of workspace calculation.
   */
  virtual OpsErrorCode CalcWorkspace(const std::vector<ir::Value> &input, const ir::Value output,
                                     size_t *workspace_size) {
    return SUCCESS;
  }

  /**
   * @brief Launch the computational kernel operation to the target device. It handles device-specific async or sync
   * execution.
   *
   * Note: If the operator needs to update output shape after launch, the shape update logic must be implemented within
   * the Launch function. Please refer to the comment for function `NeedUpdateOutputShapeAfterLaunch`
   *
   * @param input Vector of pointers to input data. Contains all input data required for computation.
   * @param workspace Vector of pointers to workspace data. Provides temporary memory for
   *                  intermediate calculations and storage during the operation.
   * @param output Pointer to the output data. Stores the result of the computation after
   *               successful execution.
   * @param stream Pointer to the device-specific execution stream (e.g., AclStream for Ascend NPU). Used for
   * asynchronous or synchronous operation. May be nullptr for synchronous CPU operations.
   * @return OpsErrorCode Return SUCCESS if execution completed successfully, or an appropriate
   *         error code if the operation failed.
   */
  virtual OpsErrorCode Launch(const std::vector<ir::Value> &input, const std::vector<ir::Value> &workspace,
                              ir::Value output, void *stream) = 0;

  /**
   * @brief This method indicates if the operator requires output shape updates after the Launch
   * function has completed execution, such as `Unique` op.
   * Note: When returning true, the shape update logic must be implemented within the Launch function itself.
   * When necessary, the Launch function needs to synchronize the stream first to ensure the kernel task execution
   * completes on the device, and then update the shape.
   *
   * @return bool Returns true if the output shape requires post-execution updates;
   *         otherwise returns false.
   */
  virtual bool NeedUpdateOutputShapeAfterLaunch() const { return false; }

 protected:
  OpName op_{Op_End};
};
}  // namespace ops
}  // namespace mrt
#endif  // __KERNEL_KERNEL_H__
