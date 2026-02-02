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

#include <vector>

#include "ops/ascend/aclnn/aclnn_contiguous.h"
#include "ops/ascend/aclnn/utils/opapi_utils.h"
#include "ops/op_register.h"
#include "hardware/ascend/res_manager/ascend_res_manager.h"

namespace mrt {
namespace ops {
OpsErrorCode AclnnContiguous::CalcWorkspace(const std::vector<const ir::Value *> &input, const ir::Value *output,
                                            size_t *workspaceSize) {
  const auto &inputTensor = input[kIndex0]->ToTensor();
  srcContiguous_ = inputTensor->IsContiguous();
  if (!srcContiguous_) {
    executor_->GetWorkspaceSize(static_cast<uint64_t *>(workspaceSize), output->ToTensor(), inputTensor);
  }
  return SUCCESS;
}

OpsErrorCode AclnnContiguous::Launch(const std::vector<const ir::Value *> &input, void *workspace, size_t workspaceSize,
                                     ir::Value *output, void *stream) {
  const auto &inputTensor = input[kIndex0]->ToTensor();
  const auto &outTensor = output->ToTensor();

  if (!srcContiguous_) {
    executor_->Launch(workspace, workspaceSize, stream, outTensor, inputTensor);
    return SUCCESS;
  }

  // Input tensor is already contiguous, perform direct memory copy
  auto srcSize = inputTensor->Numel() * inputTensor->Dtype().GetSize();
  auto dstSize = outTensor->Numel() * outTensor->Dtype().GetSize();
  if (srcSize > dstSize) {
    LOG_EXCEPTION << "Unexpected input and output size mismatch, src size is " << srcSize << ", dst size is "
                  << dstSize;
  }
  auto ret = mrt::device::ascend::AscendResManager::MemcpyDeviceToDevice(outTensor->DataPtr(), dstSize,
                                                                         inputTensor->DataPtr(), dstSize, stream);

  if (!ret) {
    LOG_ERROR << "Call aclrtMemcpyAsync in Op Contiguous failed";
    return UNKNOWN_ERROR;
  }
  return SUCCESS;
}

MRT_REG_OP(contiguous, AclnnContiguous, Ascend);
}  // namespace ops
}  // namespace mrt
