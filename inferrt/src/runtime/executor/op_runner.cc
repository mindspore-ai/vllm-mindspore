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

#include "runtime/executor/op_runner.h"

namespace mrt {
namespace runtime {
ops::OpsErrorCode OpRunner::InferShape() {
  if (isDynamicShape_) {
    return operator_->InferShape(input_, output_);
  }

  return ops::SUCCESS;
}

ops::OpsErrorCode OpRunner::CalcWorkspace() { return operator_->CalcWorkspace(input_, output_, &workspaceSize_); }

ops::OpsErrorCode OpRunner::Launch() {
  AllocateMemory();
  auto ret = operator_->Launch(input_, workspace_, workspaceSize_, output_, stream_);
  FreeMemory();
  return ret;
}

void OpRunner::AllocateMemory() {
  // TODO: adapter Ascend platform and Tuple output case.
  // Allocate memory for output tensor.
  if (output_->IsTensor()) {
    const auto &tensor = output_->ToTensor();
    if (tensor->DataPtr()) {
      LOG_EXCEPTION << "Memory leak for output of operator: " << ops::ToStr(node_->op);
    }
    auto *outputAddr = malloc(tensor->GetStorage()->SizeBytes());
    CHECK_IF_NULL(outputAddr);
    tensor->UpdateData(outputAddr);
  }

  // Allocate workspace memory if needed.
  if (workspaceSize_ > 0) {
    workspace_ = malloc(workspaceSize_);
    CHECK_IF_NULL(workspace_);
  }
}

void OpRunner::FreeMemory() {
  // TODO: adapter for Ascend platform and useless self output tensor release.
  // Free input tensors that were marked to free.
  if (!inputFreeIndex_.empty()) {
    for (auto inputIndex : inputFreeIndex_) {
      const auto &tensor = input_[inputIndex]->ToTensor();
      CHECK_IF_NULL(tensor);
      free(tensor->DataPtr());
      tensor->UpdateData(nullptr);
    }
  }

  // Free workspace memory.
  if (workspace_) {
    free(workspace_);
  }
}
}  // namespace runtime
}  // namespace mrt
