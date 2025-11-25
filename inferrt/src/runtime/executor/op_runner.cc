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
#include "common/logger.h"
#include "ops/op_def/ops_name.h"

namespace mrt {
namespace runtime {
ops::OpsErrorCode OpRunner::InferShape() {
  if (isDynamicShape_) {
    LOG_OUT << "Begin InferShape for op[" << ops::ToStr(opName_) << "], inputs=" << input_;
    return operator_->InferShape(input_, output_);
  }

  return ops::SUCCESS;
}

ops::OpsErrorCode OpRunner::CalcWorkspace() {
  LOG_OUT << "Begin CalcWorkspace for op[" << ops::ToStr(opName_) << "], inputs=" << input_ << ", output=" << *output_
          << ", workspaceSize=" << workspaceSize_;
  return operator_->CalcWorkspace(input_, output_, &workspaceSize_);
}

ops::OpsErrorCode OpRunner::Launch() {
  LOG_OUT << "Begin launch op[" << ops::ToStr(opName_) << "], inputs=" << input_ << ", workspace=" << workspace_
          << ", workspaceSize=" << workspaceSize_ << ", output=" << *output_ << ", stream=" << stream_;
  auto ret = operator_->Launch(input_, workspace_, workspaceSize_, output_, stream_);
  return ret;
}

ops::OpsErrorCode OpRunner::Launch(void *stream) {
  CHECK_IF_NULL(stream);
  LOG_OUT << "Begin launch op[" << ops::ToStr(opName_) << "], inputs=" << input_ << ", workspace=" << workspace_
          << ", workspaceSize=" << workspaceSize_ << ", output=" << *output_ << ", stream=" << stream;
  auto ret = operator_->Launch(input_, workspace_, workspaceSize_, output_, stream);
  return ret;
}

void OpRunner::AllocateMemory() {
  // Allocate memory for output tensor.
  ir::VisitAllTensors(output_, [&](const ir::TensorPtr &tensor) {
    const auto &storage = tensor->GetStorage();
    if (storage->CheckOwnsData() && storage->Data() != nullptr) {
      LOG_EXCEPTION << "Memory leak for output of operator: " << GetOpName();
    }
    // For op output ref graph input tensor case.
    bool need_alloc = storage->Data() == nullptr;
    if (need_alloc) {
      storage->AllocateMemory();
    }
  });

  // Allocate workspace memory if needed.
  if (workspaceSize_ > 0) {
    CHECK_IF_FAIL(workspace_ == nullptr);
    workspace_ = alloc_.Allocate(workspaceSize_);
    CHECK_IF_NULL(workspace_);
  }
}

void OpRunner::FreeMemory() {
  // Free input tensors that were marked to free.
  for (auto &storage : storagesToFree_) {
    storage->FreeMemory();
  }

  // Free workspace memory.
  if (workspace_) {
    alloc_.Free(workspace_);
    workspace_ = nullptr;
  }
}

void OpRunner::UpdateRefNodeOutputValue() {
  const std::vector<std::pair<uint32_t, uint32_t>> &refPairs = operator_->GetOutputInputRefPairs();
  if (refPairs.empty()) {
    return;
  }
  for (auto [outputIndex, inputIndex] : refPairs) {
    LOG_OUT << "Update op[" << GetOpName() << "] output value, outputIndex: " << outputIndex
            << ", inputIndex: " << inputIndex;
    CHECK_IF_FAIL(inputIndex < input_.size());
    auto &inputValue = input_[inputIndex];
    CHECK_IF_NULL(inputValue);
    CHECK_IF_FAIL(inputValue->IsTensor());
    auto &inputTensor = inputValue->ToTensor();
    CHECK_IF_NULL(inputTensor);

    CHECK_IF_NULL(output_);
    if (output_->IsTensor()) {
      CHECK_IF_FAIL(outputIndex == 0);
      auto &outputTensor = output_->ToTensor();
      CHECK_IF_NULL(outputTensor);
      outputTensor->SetStorage(inputTensor->GetStorage());
    } else if (output_->IsTuple()) {
      auto &outputTuple = output_->ToTuple();
      CHECK_IF_FAIL(outputIndex < outputTuple->Size());
      auto &output = (*outputTuple)[outputIndex];
      CHECK_IF_NULL(output);
      CHECK_IF_FAIL(output->IsTensor());
      auto &outputTensor = output->ToTensor();
      CHECK_IF_NULL(outputTensor);
      outputTensor->SetStorage(inputTensor->GetStorage());
    } else {
      LOG_EXCEPTION << "The output type of operator " << GetOpName()
                    << " is not supported to ref input. The output index: " << outputIndex
                    << ", input index: " << inputIndex << ", output info: " << *output_;
    }
  }
}
}  // namespace runtime
}  // namespace mrt
