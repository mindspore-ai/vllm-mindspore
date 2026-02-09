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
#include "ops/utils/utils.h"
#include "ops/op_register.h"
#include "acl/acl_rt.h"
#include "ir/tensor/tensor.h"
#include "hardware/ascend/res_manager/ascend_res_manager.h"
#include "hardware/ascend/res_manager/symbol_interface/symbol_utils.h"
#include "hardware/ascend/res_manager/symbol_interface/acl_rt_symbol.h"

namespace mrt {
namespace ops {
OpsErrorCode MemcpyOpBase::CalcWorkspace(const std::vector<const ir::Value *> &input, const ir::Value *output,
                                         size_t *workspaceSize) {
  const auto &inputTensor = input[kIndex0]->ToTensor();
  const auto &outTensor = output->ToTensor();
  if (!inputTensor->IsContiguous() || inputTensor->StorageOffset() != 0 || !IsTensorBaseFormat(inputTensor) ||
      !IsTensorBaseFormat(outTensor)) {
    LOG_EXCEPTION << "memcpy_likes operator does not support non-standard tensor memory layout, "
                  << "but got strides: " << inputTensor->Strides() << ", offset: " << inputTensor->StorageOffset()
                  << ", inputTensor format: " << FormatEnumToStr(inputTensor->Format())
                  << ", outTensor format: " << FormatEnumToStr(outTensor->Format());
  }

  return SUCCESS;
}

OpsErrorCode MemcpyOpBase::Launch(const std::vector<const ir::Value *> &input, void *workspace, size_t workspaceSize,
                                  ir::Value *output, void *stream) {
  return SUCCESS;
}

bool MemcpyOpBase::NeedLaunch() { return false; }

MRT_REG_OP(flatten, Flatten, Ascend);
MRT_REG_OP(unsqueeze, Unsqueeze, Ascend);
}  // namespace ops
}  // namespace mrt
