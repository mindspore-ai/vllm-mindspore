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
#include <string>
#include <vector>
#include "ops/ascend/hccl/hccl_reduce_scatter.h"
#include "ops/ascend/hccl/hccl_adapter.h"
#include "hardware/hardware_abstract/collective/collective_manager.h"
#include "ops/ascend/hccl/hcom_utils.h"
#include "hccl/hccl_types.h"
#include "hccl/hccl.h"

#include "common/logger.h"
#include "ops/op_register.h"

#include "hardware/ascend/res_manager/ascend_stream_manager.h"

namespace mrt {
namespace ops {
OpsErrorCode HcclReduceScatter::CalcWorkspace(const std::vector<const ir::Value *> &input, const ir::Value *output,
                                              size_t *workspaceSize) {
  LOG_OUT << "HcclReduceScatter CalcWorkspace";
  HcclAdapter::GetInstance().InitHccl();
  auto rankSize = input[kIndex2]->ToInt();
  auto [hcclCount, hcclDataType] = HcomUtil::GetHcclCountAndTypeFromTensor(input[kIndex0]->ToTensor(), rankSize);
  hcclKernel_.hcclCount_ = hcclCount;
  hcclKernel_.hcclDataType_ = hcclDataType;
  const string &groupName = input[kIndex3]->ToString();
  hcclKernel_.comm_ = HcomUtil::LoadHcclLibrary(groupName);

  return SUCCESS;
}

OpsErrorCode HcclReduceScatter::Launch(const std::vector<const ir::Value *> &input, void *workspace,
                                       size_t workspaceSize, ir::Value *output, void *stream) {
  LOG_OUT << "HcclReduceScatter launch";
  auto hcclOpType = HcomUtil::GetHcomReduceOpType(input[kIndex1]->ToString());
  auto outTensor = output->ToTensor();

  auto hcclResult = HcclAdapter::GetInstance().HcclReduceScatter(
    const_cast<void *>(input[kIndex0]->ToTensor()->DataPtr()), outTensor->DataPtr(), hcclKernel_.hcclCount_,
    hcclKernel_.hcclDataType_, hcclOpType, stream, hcclKernel_.comm_);

  if (hcclResult != ::HcclResult::HCCL_SUCCESS) {
    LOG_ERROR << "HcclReduceScatter failed, hccl_result: " << hcclResult;
  }

  return SUCCESS;
}
MRT_REG_OP(reduce_scatter, HcclReduceScatter, Ascend);
}  // namespace ops
}  // namespace mrt
