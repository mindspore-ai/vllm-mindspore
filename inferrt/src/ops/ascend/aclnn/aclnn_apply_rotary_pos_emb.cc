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

#include "ops/ascend/aclnn/aclnn_apply_rotary_pos_emb.h"
#include "ops/ascend/aclnn/utils/opapi_utils.h"
#include "ops/op_register.h"

static const int64_t ROPE_Layout_BSND = 1;

namespace mrt {
namespace ops {
OpsErrorCode AclnnApplyRotaryPosEmb::CalcWorkspace(const std::vector<const ir::Value *> &input, const ir::Value *output,
                                                   size_t *workspaceSize) {
  LOG_OUT << "Begin CalcWorkspace for op [apply_rotary_pos_emb]";
  executor_->GetWorkspaceSize(static_cast<uint64_t *>(workspaceSize), input[kIndex0]->ToTensor(),
                              input[kIndex1]->ToTensor(), input[kIndex2]->ToTensor(), input[kIndex3]->ToTensor(),
                              ROPE_Layout_BSND);
  return SUCCESS;
}

OpsErrorCode AclnnApplyRotaryPosEmb::Launch(const std::vector<const ir::Value *> &input, void *workspace,
                                            size_t workspaceSize, ir::Value *output, void *stream) {
  LOG_OUT << "Begin Launch op [apply_rotary_pos_emb]";
  executor_->Launch(workspace, workspaceSize, stream, input[kIndex0]->ToTensor(), input[kIndex1]->ToTensor(),
                    input[kIndex2]->ToTensor(), input[kIndex3]->ToTensor(), ROPE_Layout_BSND);
  return SUCCESS;
}

std::vector<std::pair<uint32_t, uint32_t>> AclnnApplyRotaryPosEmb::GetOutputInputRefPairs() const {
  return {{0, 0}, {1, 1}};
}

MRT_REG_OP(apply_rotary_pos_emb, AclnnApplyRotaryPosEmb, Ascend);
}  // namespace ops
}  // namespace mrt
