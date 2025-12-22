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

#include "ops/ascend/lowered/op_linalg_call.h"

#include <functional>

#include "common/logger.h"
#include "ops/ascend/lowered/lowered_op_helper.h"

namespace mrt {
namespace ops {

// Index constants following OpCustomCall naming convention
constexpr size_t kInputMlirTextIndex = 0;   // MLIR text at position 0
constexpr size_t kRealInputStartIndex = 1;  // Actual inputs start from position 1

void OpLinalgCall::Init(const std::vector<const ir::Value *> &inputs, const ir::Value *output) {
  (void)output;
  // 1. Extract MLIR text (first parameter)
  CHECK_IF_NULL(inputs[kInputMlirTextIndex]);
  mlirText_ = inputs[kInputMlirTextIndex]->ToString();
  if (mlirText_.empty()) {
    LOG_EXCEPTION << "MLIR text is empty in linalg_call";
  }

  // 2. Create operator instance using LoweredOpHelper
  loweredOp_ = LoweredOpHelper::CreateFromMlirText(mlirText_);
  if (loweredOp_ == nullptr) {
    LOG_EXCEPTION << "Failed to create Linalg operator from MLIR text. "
                  << "Check if MLIR contains hacc.entry annotation.";
  }

  // 3. Extract actual inputs (excluding the first MLIR text parameter)
  auto inputSize = inputs.size() - kRealInputStartIndex;
  realInputs_.resize(inputSize, nullptr);
  for (size_t i = kRealInputStartIndex; i < inputs.size(); i++) {
    realInputs_[i - kRealInputStartIndex] = inputs[i];
  }

  // Note: AutoLoweredOp does not need explicit Init() call, initialization is done during construction
  LOG_OUT << "OpLinalgCall initialized with MLIR text hash: " << std::hash<std::string>{}(mlirText_);
}

OpsErrorCode OpLinalgCall::InferShape(const std::vector<const ir::Value *> &input, ir::Value *output) {
  (void)input;
  if (loweredOp_ == nullptr) {
    LOG_ERROR << "loweredOp_ is null in OpLinalgCall::InferShape";
    return UNKNOWN_ERROR;
  }
  // Pass realInputs_ instead of input: input contains MLIR text, loweredOp_ only needs actual tensor inputs
  auto ret = loweredOp_->InferShape(realInputs_, output);
  if (ret != OpsErrorCode::SUCCESS) return ret;
  // Ensure symbolic shapes (if attached) are evaluated before downstream ops / launch.
  return Operator::InferShape(realInputs_, output);
}

OpsErrorCode OpLinalgCall::CalcWorkspace(const std::vector<const ir::Value *> &input, const ir::Value *output,
                                         size_t *workspaceSize) {
  (void)input;
  if (loweredOp_ == nullptr) {
    LOG_ERROR << "loweredOp_ is null in OpLinalgCall::CalcWorkspace";
    return UNKNOWN_ERROR;
  }
  return loweredOp_->CalcWorkspace(realInputs_, output, workspaceSize);
}

OpsErrorCode OpLinalgCall::Launch(const std::vector<const ir::Value *> &input, void *workspace, size_t workspaceSize,
                                  ir::Value *output, void *stream) {
  (void)input;
  if (loweredOp_ == nullptr) {
    LOG_ERROR << "loweredOp_ is null in OpLinalgCall::Launch";
    return UNKNOWN_ERROR;
  }
  return loweredOp_->Launch(realInputs_, workspace, workspaceSize, output, stream);
}

}  // namespace ops
}  // namespace mrt
