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

#include "ops/ascend/dvm/dvm_op.h"
#include "common/logger.h"

namespace mrt::ops {

DvmOp::DvmOp(dvm::KernelType kernelType, BuildFunc buildFunc)
    : kernelType_(kernelType), buildFunc_(std::move(buildFunc)), executor_(nullptr), isInitialized_(false) {
  if (!buildFunc_) {
    LOG_EXCEPTION << "Build function is null";
  }

  // Create executor
  executor_ = std::make_unique<DvmKernelExecutor>(kernelType_);

  LOG_OUT << "DvmOp created with kernel type: " << static_cast<int>(kernelType_);
}

void DvmOp::Init(const std::vector<const ir::Value *> &inputs, const ir::Value *output) {
  if (isInitialized_) {
    LOG_OUT << "DvmOp already initialized, skipping";
    return;
  }

  if (executor_ == nullptr) {
    LOG_EXCEPTION << "Executor is null in Init";
  }

  LOG_OUT << "DvmOp::Init - building kernel graph";

  // Build kernel graph using user-provided function
  int ret = executor_->BuildKernel(buildFunc_, inputs, output);
  if (ret != 0) {
    LOG_EXCEPTION << "Failed to build DVM kernel in Init, error code: " << ret;
  }

  isInitialized_ = true;
  LOG_OUT << "DvmOp::Init completed successfully";
}

OpsErrorCode DvmOp::CalcWorkspace(const std::vector<const ir::Value *> &input, const ir::Value *output,
                                  size_t *workspaceSize) {
  if (executor_ == nullptr) {
    LOG_ERROR << "Executor is null in CalcWorkspace";
    return UNKNOWN_ERROR;
  }

  if (workspaceSize == nullptr) {
    LOG_ERROR << "workspaceSize pointer is null";
    return INVALID_PARAM;
  }

  // Ensure kernel is built
  if (!isInitialized_) {
    LOG_OUT << "DvmOp not initialized, calling Init";
    Init(input, output);
  }

  LOG_OUT << "DvmOp::CalcWorkspace - calling executor";

  // Delegate to executor
  int ret = executor_->GetWorkspaceSize(workspaceSize, input, output);
  if (ret != 0) {
    LOG_ERROR << "DvmKernelExecutor::GetWorkspaceSize failed with code: " << ret;
    return UNKNOWN_ERROR;
  }

  LOG_OUT << "DvmOp::CalcWorkspace - workspace size: " << *workspaceSize;
  return SUCCESS;
}

OpsErrorCode DvmOp::Launch(const std::vector<const ir::Value *> &input, void *workspace, size_t workspaceSize,
                           ir::Value *output, void *stream) {
  if (executor_ == nullptr) {
    LOG_ERROR << "Executor is null in Launch";
    return UNKNOWN_ERROR;
  }

  if (!isInitialized_) {
    LOG_ERROR << "DvmOp not initialized, call CalcWorkspace first";
    return UNKNOWN_ERROR;
  }

  LOG_OUT << "DvmOp::Launch - calling executor";

  // Delegate to executor
  int ret = executor_->Launch(workspace, workspaceSize, stream, input, output);
  if (ret != 0) {
    LOG_ERROR << "DvmKernelExecutor::Launch failed with code: " << ret;
    return UNKNOWN_ERROR;
  }

  LOG_OUT << "DvmOp::Launch - success";
  return SUCCESS;
}

}  // namespace mrt::ops
