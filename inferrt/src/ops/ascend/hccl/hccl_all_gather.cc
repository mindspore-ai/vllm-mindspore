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
#include <string>

#include "ops/ascend/hccl/hccl_all_gather.h"
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
OpsErrorCode HcclAllGather::CalcWorkspace(const std::vector<const ir::Value *> &input, const ir::Value *output,
                                          size_t *workspace_size) {
  LOG_OUT << "HcclAllGather CalcWorkspace";
  auto rank_id = mrt::collective::CollectiveManager::Instance().local_rank_id();
  std::string rank_id_str = std::to_string(0);
  HcclAdapter::GetInstance().InitHccl(rank_id, rank_id_str);
  auto [hccl_count, hccl_data_type] = HcomUtil::GetHcclCountAndTypeFromTensor(input[kIndex0]->ToTensor());
  hcclKernel.hccl_count_ = hccl_count;
  hcclKernel.hccl_data_type_ = hccl_data_type;
  const string &group_name = input[kIndex2]->ToString();
  hcclKernel.comm_ = HcomUtil::LoadHcclLibrary(group_name);

  return SUCCESS;
}

OpsErrorCode HcclAllGather::Launch(const std::vector<const ir::Value *> &input, void *workspace, size_t workspaceSize,
                                   ir::Value *output, void *stream) {
  LOG_OUT << "HcclAllGather launch";

  auto hccl_result = HcclAdapter::GetInstance().HcclAllGather(const_cast<void *>(input[kIndex0]->ToTensor()->DataPtr()),
                                                              output->ToTensor()->DataPtr(), hcclKernel.hccl_count_,
                                                              hcclKernel.hccl_data_type_, stream, hcclKernel.comm_);
  if (hccl_result != ::HcclResult::HCCL_SUCCESS) {
    LOG_ERROR << "HcomAllGather failed, hccl_result: " << hccl_result;
  }

  return SUCCESS;
}
MRT_REG_OP(all_gather, HcclAllGather, Ascend);
}  // namespace ops
}  // namespace mrt
