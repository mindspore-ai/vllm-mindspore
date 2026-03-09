/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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

#include "ops/ascend/aclnn/aclnn_scatter_nd_update.h"
#include <vector>
#include "hardware/ascend/res_manager/ascend_res_manager.h"
#include "ops/ascend/aclnn/utils/opapi_utils.h"
#include "ops/op_register.h"

namespace mrt {
namespace ops {
OpsErrorCode AclnnScatterNdUpdateNonInplace::CalcWorkspace(const std::vector<const ir::Value *> &input,
                                                           const ir::Value *output, size_t *workspaceSize) {
  executor_->GetWorkspaceSize(static_cast<uint64_t *>(workspaceSize), output->ToTensor(), input[1]->ToTensor(),
                              input[2]->ToTensor());
  return SUCCESS;
}

OpsErrorCode AclnnScatterNdUpdateNonInplace::Launch(const std::vector<const ir::Value *> &input, void *workspace,
                                                    size_t workspaceSize, ir::Value *output, void *stream) {
  const auto &dstTensor = output->ToTensor();
  const auto &srcTensor = input[0]->ToTensor();
  auto dstSize = dstTensor->Numel() * dstTensor->Dtype().GetSize();
  auto srcSize = srcTensor->Numel() * srcTensor->Dtype().GetSize();
  if (srcSize > dstSize) {
    LOG_EXCEPTION << "Unexpected input and output size mismatch for scatter_nd_update, src size is " << srcSize
                  << ", dst size is " << dstSize;
  }
  device::ascend::AscendResManager::MemcpyDeviceToDevice(dstTensor->DataPtr(), dstSize, srcTensor->DataPtr(), srcSize,
                                                         stream);
  executor_->Launch(workspace, workspaceSize, stream, output->ToTensor(), input[1]->ToTensor(), input[2]->ToTensor());
  return SUCCESS;
}

MRT_REG_OP(scatter_nd_update, AclnnScatterNdUpdateNonInplace, Ascend);
}  // namespace ops
}  // namespace mrt
