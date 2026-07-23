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

#include "ops/ascend/hccl/hccl_broadcast.h"

#include <string>
#include <vector>

#include "common/logger.h"
#include "hardware/ascend/res_manager/ascend_res_manager.h"
#include "hccl/hccl.h"
#include "hccl/hccl_types.h"
#include "ops/ascend/hccl/hccl_adapter.h"
#include "ops/ascend/hccl/hcom_utils.h"
#include "ops/op_register.h"

namespace mrt {
namespace ops {
OpsErrorCode HcclBroadcast::CalcWorkspace(const std::vector<const ir::Value *> &input, const ir::Value *output,
                                          size_t *workspaceSize) {
  HcclAdapter::GetInstance().InitHccl();
  auto inputTensor = input[kIndex0]->ToTensor();
  HcomUtil::CheckHcclInputContiguous(inputTensor, "HcclBroadcast");
  auto [hcclCount, hcclDataType] = HcomUtil::GetHcclCountAndTypeFromTensor(inputTensor);
  hcclKernel_.hcclCount_ = hcclCount;
  hcclKernel_.hcclDataType_ = hcclDataType;
  const std::string &groupName = input[kIndex2]->ToString();
  hcclKernel_.comm_ = HcomUtil::LoadHcclLibrary(groupName);
  return SUCCESS;
}

OpsErrorCode HcclBroadcast::Launch(const std::vector<const ir::Value *> &input, void *workspace, size_t workspaceSize,
                                   ir::Value *output, void *stream) {
  auto inputTensor = input[kIndex0]->ToTensor();
  auto outTensor = output->ToTensor();
  auto dstSize = outTensor->Numel() * outTensor->Dtype().GetSize();

  auto copyRet = mrt::device::ascend::AscendResManager::MemcpyDeviceToDevice(outTensor->DataPtr(), dstSize,
                                                                             inputTensor->DataPtr(), dstSize, stream);
  if (!copyRet) {
    RT_GLOG(ERROR) << "HcclBroadcast copy input to output failed";
    return LAUNCH_OP_FAILED;
  }

  auto root = static_cast<uint32_t>(input[kIndex1]->ToInt());
  auto hcclResult = HcclAdapter::GetInstance().HcclBroadcast(
    outTensor->DataPtr(), hcclKernel_.hcclCount_, hcclKernel_.hcclDataType_, root, stream, hcclKernel_.comm_);
  if (hcclResult != ::HcclResult::HCCL_SUCCESS) {
    RT_GLOG(ERROR) << "HcclBroadcast failed, hcclResult: " << hcclResult;
    return LAUNCH_OP_FAILED;
  }
  return SUCCESS;
}

MRT_REG_OP(broadcast, HcclBroadcast, Ascend);
}  // namespace ops
}  // namespace mrt
