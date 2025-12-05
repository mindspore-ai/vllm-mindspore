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

#include "ops/ascend/mem/memcpy_likes.h"
#include "ops/ascend/aclnn/utils/opapi_utils.h"
#include "ops/op_register.h"
#include "acl/acl_rt.h"
#include "hardware/ascend/res_manager/ascend_res_manager.h"
#include "hardware/ascend/res_manager/symbol_interface/symbol_utils.h"
#include "hardware/ascend/res_manager/symbol_interface/acl_rt_symbol.h"

namespace mrt {
namespace ops {
OpsErrorCode ViewMemcpyOpBase::CalcWorkspace(const std::vector<const ir::Value *> &input, const ir::Value *output,
                                             size_t *workspaceSize) {
  return SUCCESS;
}

OpsErrorCode ViewMemcpyOpBase::Launch(const std::vector<const ir::Value *> &input, void *workspace,
                                      size_t workspaceSize, ir::Value *output, void *stream) {
  auto inputTensor = input[kIndex0]->ToTensor();
  auto srcSize = inputTensor->Numel() * inputTensor->Dtype().GetSize();
  auto outTensor = output->ToTensor();
  auto dstSize = outTensor->Numel() * outTensor->Dtype().GetSize();
  if (srcSize > dstSize) {
    LOG_EXCEPTION << "unexpected input and output, src size is " << srcSize << ", dst size is " << dstSize;
  }

  if (outTensor->DataPtr() != inputTensor->DataPtr()) {
    auto ret = mrt::device::ascend::AscendResManager::MemcpyDeviceToDevice(outTensor->DataPtr(), dstSize,
                                                                           inputTensor->DataPtr(), dstSize, stream);
    if (ret == false) {
      LOG_ERROR << " call aclrtMemcpyAsync in Op TensorCopy failed";
      return UNKNOWN_ERROR;
    }
  }

  return SUCCESS;
}

MRT_REG_OP(flatten, ViewFlatten, Ascend);
}  // namespace ops
}  // namespace mrt
