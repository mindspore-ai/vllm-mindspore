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

#include "ops/ascend/hccl/wait_tensor.h"
#include "ops/ascend/hccl/hccl_adapter.h"
#include "ops/ascend/hccl/hcom_utils.h"
#include "hccl/hccl_types.h"
#include "hccl/hccl.h"

#include "common/logger.h"
#include "ops/op_register.h"
#include "hardware/ascend/res_manager/ascend_stream_manager.h"
#include "hardware/ascend/res_manager/ascend_res_manager.h"
namespace mrt {
namespace ops {

OpsErrorCode HcclWaitTensor::InferShape(const std::vector<const ir::Value *> &input, ir::Value *output) {
  LOG_OUT << "WaitTensor InferShape";
  auto &input0Shape = input[kIndex0]->ToTensor()->Shape();
  auto &outputTensor = output->ToTensor();
  auto &outputShape = outputTensor->Shape();
  outputShape = input0Shape;
  outputTensor->Resize();
  return SUCCESS;
}

OpsErrorCode HcclWaitTensor::CalcWorkspace(const std::vector<const ir::Value *> &input, const ir::Value *output,
                                           size_t *workspace_size) {
  return SUCCESS;
}

OpsErrorCode HcclWaitTensor::Launch(const std::vector<const ir::Value *> &input, void *workspace, size_t workspaceSize,
                                    ir::Value *output, void *stream) {
  LOG_OUT << "WaitTensor launch";

  auto src_tensor = input[kIndex0]->ToTensor();
  auto out_tensor = output->ToTensor();
  auto dst_size = out_tensor->Numel() * out_tensor->Dtype().GetSize();

  auto ret = mrt::device::ascend::AscendResManager::MemcpyDeviceToDevice(out_tensor->DataPtr(), dst_size,
                                                                         src_tensor->DataPtr(), dst_size, stream);
  if (ret == false) {
    LOG_ERROR << " call aclrtMemcpyAsync in Op HcclTensorCopy failed";
  }

  mrt::device::ascend::AscendStreamMng::GetInstance().SyncStream(stream);
  return SUCCESS;
}
MRT_REG_OP(wait_tensor, HcclWaitTensor, Ascend);
}  // namespace ops
}  // namespace mrt
