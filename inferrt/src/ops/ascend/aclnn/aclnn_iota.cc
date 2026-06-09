/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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

#include "ops/ascend/aclnn/aclnn_iota.h"

#include "ops/op_register.h"

namespace mrt {
namespace ops {
namespace {
constexpr size_t kLengthIdx = 0;
constexpr size_t kStartIdx = 1;
constexpr size_t kStepIdx = 2;
constexpr size_t kIotaInputNum = 3;

int64_t CalcIotaEnd(const std::vector<const ir::Value *> &input) {
  CHECK_IF_FAIL(input.size() == kIotaInputNum);
  const auto length = input[kLengthIdx]->ToInt();
  const auto start = input[kStartIdx]->ToInt();
  const auto step = input[kStepIdx]->ToInt();
  return start + length * step;
}
}  // namespace

OpsErrorCode AclnnIota::CalcWorkspace(const std::vector<const ir::Value *> &input, const ir::Value *output,
                                      size_t *workspaceSize) {
  end_ = ir::Value(CalcIotaEnd(input));
  executor_->GetWorkspaceSize(static_cast<uint64_t *>(workspaceSize), input[kStartIdx],
                              static_cast<const ir::Value *>(&end_), input[kStepIdx], output->ToTensor());
  return SUCCESS;
}

OpsErrorCode AclnnIota::Launch(const std::vector<const ir::Value *> &input, void *workspace, size_t workspaceSize,
                               ir::Value *output, void *stream) {
  executor_->Launch(workspace, workspaceSize, stream, input[kStartIdx], static_cast<const ir::Value *>(&end_),
                    input[kStepIdx], output->ToTensor());
  return SUCCESS;
}

MRT_REG_OP(iota, AclnnIota, Ascend);
}  // namespace ops
}  // namespace mrt
