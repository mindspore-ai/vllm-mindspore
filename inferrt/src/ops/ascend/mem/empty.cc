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

#include "ops/ascend/mem/empty.h"
#include "ops/ascend/aclnn/utils/opapi_utils.h"
#include "ops/op_register.h"

namespace mrt {
namespace ops {
OpsErrorCode Empty::CalcWorkspace(const std::vector<const ir::Value *> &input, const ir::Value *output,
                                  size_t *workspaceSize) {
  CHECK_IF_FAIL(input.size() >= kInputSize1);
  return SUCCESS;
}

OpsErrorCode Empty::Launch(const std::vector<const ir::Value *> &input, void *workspace, size_t workspaceSize,
                           ir::Value *output, void *stream) {
  CHECK_IF_FAIL(input.size() >= kInputSize1);
  return SUCCESS;
}

MRT_REG_OP(empty, Empty, Ascend);
}  // namespace ops
}  // namespace mrt
