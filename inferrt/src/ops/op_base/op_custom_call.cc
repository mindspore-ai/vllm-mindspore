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

#include "ops/op_base/op_custom_call.h"
#include "ops/custom_op_register.h"
#ifdef ENABLE_TORCH_FRONT
#include "ops/op_base/op_torch_call.h"
#endif

namespace mrt {
namespace ops {
constexpr size_t kInputIOpNameIndex = 0;
constexpr size_t kRealInputIndex = 1;

void OpCustomCall::Init(const std::vector<const ir::Value *> &inputs, const ir::Value *output) {
  CHECK_IF_NULL(inputs[kInputIOpNameIndex]);
  opName_ = inputs[kInputIOpNameIndex]->ToString();
  size_t pos = opName_.find(".");
  if (pos == std::string::npos) {
    LOG_EXCEPTION << "Invalid op name: " << opName_ << ". Op name must be in the format of ns.op_name.";
  }
  std::string opName = opName_.substr(pos + 1);
  operatorPtr_ = CreateCustomOperator(opName);
  SetOpType(OpType::CustomCallOp);
#ifdef ENABLE_TORCH_FRONT
  if (operatorPtr_ == nullptr) {
    LOG_OUT << "Custom op " << opName_ << " not registered. Try to create operator from torch.";
    operatorPtr_ = std::make_shared<OpTorchCall>(opName_);
    SetOpType(OpType::TorchCallOp);
  }
#endif
  CHECK_IF_NULL(operatorPtr_);
  operatorPtr_->Init(inputs, output);
  auto inputSize = inputs.size() - kRealInputIndex;
  input_.resize(inputSize, nullptr);
  for (size_t i = kRealInputIndex; i < inputs.size(); i++) {
    input_[i - kRealInputIndex] = inputs[i];
  }
}

OpsErrorCode OpCustomCall::InferShape(const std::vector<const ir::Value *> &input, ir::Value *output) {
  if (operatorPtr_ == nullptr) {
    LOG_ERROR << "operatorPtr_ is null in OpCustomCall::InferShape";
    return UNKNOWN_ERROR;
  }
  return operatorPtr_->InferShape(input_, output);
}

OpsErrorCode OpCustomCall::CalcWorkspace(const std::vector<const ir::Value *> &input, const ir::Value *output,
                                         size_t *workspaceSize) {
  return operatorPtr_->CalcWorkspace(input_, output, workspaceSize);
}

OpsErrorCode OpCustomCall::Launch(const std::vector<const ir::Value *> &input, void *workspace, size_t workspaceSize,
                                  ir::Value *output, void *stream) {
  return operatorPtr_->Launch(input_, workspace, workspaceSize, output, stream);
}

}  // namespace ops
}  // namespace mrt
