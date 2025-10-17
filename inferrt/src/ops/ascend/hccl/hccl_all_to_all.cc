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

#include "ops/ascend/hccl/hccl_all_to_all.h"
#include "hardware/hardware_abstract/collective/collective_manager.h"
#include "hardware/ascend/res_manager/ascend_stream_manager.h"
#include "ops/ascend/hccl/hcom_utils.h"
#include "ops/ascend/hccl/hccl_adapter.h"
#include "hccl/hccl_types.h"
#include "hccl/hccl.h"

#include "common/logger.h"
#include "ops/op_register.h"

namespace mrt {
namespace ops {
bool is_all_to_all_v(const ir::TuplePtr &sendNumelList, const ir::TuplePtr &recvNumelList) {
  for (size_t i = 0; i < sendNumelList->Size(); i++) {
    if (sendNumelList->operator[](i)->ToInt() != sendNumelList->operator[](0)->ToInt()) {
      return true;
    }
  }
  for (size_t i = 0; i < recvNumelList->Size(); i++) {
    if (recvNumelList->operator[](i)->ToInt() != recvNumelList->operator[](0)->ToInt()) {
      return true;
    }
  }
  return false;
}

void GetAllToAllVParam(const ir::TuplePtr &sendNumelList, const ir::TuplePtr &recvNumelList,
                       HcclAllToAllVParams *params) {
  uint64_t offset = 0;
  for (size_t i = 0; i < sendNumelList->Size(); i++) {
    auto count = static_cast<uint64_t>(sendNumelList->operator[](i)->ToInt());
    params->sendCounts.push_back(count);
    params->sdispls.push_back(offset);
    offset += count;
  }
  offset = 0;
  for (size_t i = 0; i < recvNumelList->Size(); i++) {
    auto count = static_cast<uint64_t>(recvNumelList->operator[](i)->ToInt());
    params->recvCounts.push_back(count);
    params->rdispls.push_back(offset);
    offset += count;
  }
}

OpsErrorCode HcclAllToAll::CalcWorkspace(const std::vector<const ir::Value *> &input, const ir::Value *output,
                                         size_t *workspaceSize) {
  LOG_OUT << "HcclAllToAll CalcWorkspace";
  const string &groupName = input[kIndex3]->ToString();
  auto rankSize = mrt::collective::CollectiveManager::Instance().GetGroupSize(groupName);
  HcclAdapter::GetInstance().InitHccl();
  auto [hcclCount, hcclDataType] = HcomUtil::GetHcclCountAndTypeFromTensor(input[kIndex0]->ToTensor());
  hcclKernel_.hcclCount_ = hcclCount / rankSize;
  hcclKernel_.hcclDataType_ = hcclDataType;
  hcclKernel_.comm_ = HcomUtil::LoadHcclLibrary(groupName);
  useAllToAllV_ = is_all_to_all_v(input[kIndex2]->ToTuple(), input[kIndex1]->ToTuple());
  return SUCCESS;
}

OpsErrorCode HcclAllToAll::Launch(const std::vector<const ir::Value *> &input, void *workspace, size_t workspaceSize,
                                  ir::Value *output, void *stream) {
  LOG_OUT << "HcclAllToAll launch";
  auto outTensor = output->ToTensor();
  ::HcclResult hcclResult;
  if (useAllToAllV_) {
    LOG_OUT << "HcclAllToAll launch AllToAllV Kernel";
    HcclAllToAllVParams params;
    GetAllToAllVParam(input[kIndex2]->ToTuple(), input[kIndex1]->ToTuple(), &params);
    hcclResult = HcclAdapter::GetInstance().HcclAlltoAllV(const_cast<void *>(input[kIndex0]->ToTensor()->DataPtr()),
                                                           outTensor->DataPtr(), params, hcclKernel_.hcclDataType_,
                                                           stream, hcclKernel_.comm_);
  } else {
    LOG_OUT << "HcclAllToAll launch AllToAll Kernel";
    HcclAllToAllParams params = {hcclKernel_.hcclCount_, hcclKernel_.hcclCount_};
    hcclResult = HcclAdapter::GetInstance().HcclAllToAll(const_cast<void *>(input[kIndex0]->ToTensor()->DataPtr()),
                                                          outTensor->DataPtr(), params, hcclKernel_.hcclDataType_,
                                                          stream, hcclKernel_.comm_);
  }

  if (hcclResult != ::HcclResult::HCCL_SUCCESS) {
    LOG_ERROR << "HcclAllToAll failed, hcclResult: " << hcclResult;
  }

  return SUCCESS;
}
MRT_REG_OP(all_to_all, HcclAllToAll, Ascend);
}  // namespace ops
}  // namespace mrt
