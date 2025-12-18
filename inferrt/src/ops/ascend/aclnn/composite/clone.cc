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
#include "ops/ascend/aclnn/composite/clone.h"
#include "ops/ascend/aclnn/utils/opapi_utils.h"
#include "ops/op_register.h"

namespace mrt {
namespace ops {
OpsErrorCode AclnnClone::CalcWorkspace(const std::vector<const ir::Value *> &input, const ir::Value *output,
                                       size_t *workspaceSize) {
  executor_->GetWorkspaceSize(static_cast<uint64_t *>(workspaceSize), output->ToTensor(), input[kIndex0]->ToTensor());
  return SUCCESS;
}

OpsErrorCode AclnnClone::Launch(const std::vector<const ir::Value *> &input, void *workspace, size_t workspaceSize,
                                ir::Value *output, void *stream) {
  executor_->Launch(workspace, workspaceSize, stream, output->ToTensor(), input[kIndex0]->ToTensor());
  return SUCCESS;
}

MRT_REG_OP(clone, AclnnClone, Ascend);
}  // namespace ops
}  // namespace mrt
