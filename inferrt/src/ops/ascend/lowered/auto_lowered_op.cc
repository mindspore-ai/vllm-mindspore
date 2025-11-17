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

#include "ops/ascend/lowered/auto_lowered_op.h"
#include "common/logger.h"

namespace mrt::ops {

AutoLoweredOp::AutoLoweredOp(const std::string &specId) : specId_(specId), executor_(nullptr) {
  executor_ = std::make_unique<LoweredKernelExecutor>(specId);
  LOG_OUT << "AutoLoweredOp created for spec: " << specId;
}

OpsErrorCode AutoLoweredOp::CalcWorkspace(const std::vector<const ir::Value *> &input, const ir::Value *output,
                                          size_t *workspaceSize) {
  if (executor_ == nullptr) {
    LOG_ERROR << "Executor is null in CalcWorkspace";
    return UNKNOWN_ERROR;
  }

  if (workspaceSize == nullptr) {
    LOG_ERROR << "workspaceSize pointer is null";
    return INVALID_PARAM;
  }

  LOG_OUT << "AutoLoweredOp::CalcWorkspace - calling executor";

  // Delegate to executor
  int ret = executor_->GetWorkspaceSize(workspaceSize, input, output);
  if (ret != 0) {
    LOG_ERROR << "LoweredKernelExecutor::GetWorkspaceSize failed with code: " << ret;
    return UNKNOWN_ERROR;
  }

  LOG_OUT << "AutoLoweredOp::CalcWorkspace - workspace size: " << *workspaceSize;
  return SUCCESS;
}

OpsErrorCode AutoLoweredOp::Launch(const std::vector<const ir::Value *> &input, void *workspace, size_t workspaceSize,
                                   ir::Value *output, void *stream) {
  if (executor_ == nullptr) {
    LOG_ERROR << "Executor is null in Launch";
    return UNKNOWN_ERROR;
  }

  LOG_OUT << "AutoLoweredOp::Launch - calling executor";

  // Delegate to executor
  int ret = executor_->Launch(workspace, workspaceSize, stream, input, output);
  if (ret != 0) {
    LOG_ERROR << "LoweredKernelExecutor::Launch failed with code: " << ret;
    return UNKNOWN_ERROR;
  }

  LOG_OUT << "AutoLoweredOp::Launch - success";
  return SUCCESS;
}

}  // namespace mrt::ops
