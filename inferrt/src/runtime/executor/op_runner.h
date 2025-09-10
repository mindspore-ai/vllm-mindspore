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

#ifndef __RUNTIME_EXECUTOR_OP_RUNNER_H_
#define __RUNTIME_EXECUTOR_OP_RUNNER_H_

#include <vector>

#include "ops/operator.h"
#include "ir/graph.h"
#include "ir/value/value.h"

namespace mrt {
namespace runtime {
/**
 * @brief OpRunner class is responsible for executing operator in the runtime.
 * It manages the execution of an operator including shape inference,
 * workspace calculation, and actual launch.
 */
class OpRunner {
 public:
  OpRunner() = default;

  OpRunner(ir::Node *node, std::unique_ptr<ops::Operator> &&operatorPtr, void *stream, bool isDynamicShape)
      : stream_(stream), node_(node), operator_(std::move(operatorPtr)), isDynamicShape_(isDynamicShape) {
    output_ = node->output.get();
    auto inputSize = node->inputs.size();
    input_.resize(inputSize, nullptr);
    for (size_t i = 0; i < inputSize; ++i) {
      auto &inputNodePtr = node->inputs[i];
      input_[i] = inputNodePtr != nullptr ? inputNodePtr->output.get() : nullptr;
    }
  }

  OpRunner(OpRunner &&other) noexcept
      : inputFreeIndex_(std::move(other.inputFreeIndex_)),
        input_(std::move(other.input_)),
        workspace_(other.workspace_),
        workspaceSize_(other.workspaceSize_),
        output_(other.output_),
        stream_(other.stream_),
        node_(other.node_),
        operator_(std::move(other.operator_)),
        isDynamicShape_(other.isDynamicShape_) {}

  OpRunner &operator=(OpRunner &&other) noexcept {
    if (this != &other) {
      inputFreeIndex_ = std::move(other.inputFreeIndex_);
      input_ = std::move(other.input_);
      workspace_ = other.workspace_;
      workspaceSize_ = other.workspaceSize_;
      output_ = other.output_;
      stream_ = other.stream_;
      node_ = other.node_;
      operator_ = std::move(other.operator_);
      isDynamicShape_ = other.isDynamicShape_;
    }
    return *this;
  }

  OpRunner(const OpRunner &) = delete;

  OpRunner &operator=(const OpRunner &) = delete;

  ~OpRunner() = default;

  /**
   * @brief Performs shape inference for the operator if it has dynamic shapes.
   * @return Operator error code indicating success or failure.
   */
  ops::OpsErrorCode InferShape();

  /**
   * @brief Calculates the required workspace size for the operator.
   * @return Operatoir error code indicating success or failure.
   */
  ops::OpsErrorCode CalcWorkspace();

  /**
   * @brief Launches the operator execution.
   * @return Operatoir error code indicating success or failure.
   */
  ops::OpsErrorCode Launch();

  /**
   * @brief Gets the IR node associated with this operator runner.
   * @return Pointer to the IR node.
   */
  const ir::Node *GetNode() const { return node_; }

  /**
   * @brief Sets the indices of input tensors that should be freed after execution.
   * @param inputFreeIndex Vector of indices indicating which input tensors to free.
   */
  void SetInputFreeIndex(std::vector<size_t> &&inputFreeIndex) { inputFreeIndex_ = std::move(inputFreeIndex); }

 private:
  /**
   * @brief Allocate device memory for output tensor and workspace.
   */
  void AllocateMemory();

  /**
   * @brief Free device memory for input tensors and workspace.
   */
  void FreeMemory();

  // Indices of input tensors that should be freed after operator execution.
  std::vector<size_t> inputFreeIndex_;

  // Input values for the operator.
  std::vector<const ir::Value *> input_;

  // Workspace memory pointer for the operator.
  void *workspace_{nullptr};

  // Size of the total workspace memory in bytes.
  size_t workspaceSize_{0};

  // Output value of the operator.
  ir::Value *output_;

  // Execution stream (e.g., Ascend stream) for the operator.
  void *stream_;

  // The allocator used for device memory management for workspace.
  Allocator alloc_;

  // IR node representing the operator to be executed.
  ir::Node *node_;

  // The operator to implementation.
  std::unique_ptr<ops::Operator> operator_;

  // Flag indicating if the operator has dynamic shapes.
  bool isDynamicShape_;
};
}  // namespace runtime
}  // namespace mrt

#endif  // __RUNTIME_EXECUTOR_OP_RUNNER_H_
