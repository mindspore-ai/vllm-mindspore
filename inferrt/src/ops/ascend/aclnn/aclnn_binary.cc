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

#include "ops/ascend/aclnn/aclnn_binary.h"
#include "ops/op_register.h"
#include "ops/op_base/utils.h"

namespace mrt {
namespace ops {
OpsErrorCode AclnnBinaryOpBase::InferShape(const std::vector<const ir::Value *> &input, ir::Value *output) {
  CHECK_IF_FAIL(input.size() == kInputSize2);
  const auto &input0 = input[kIndex0]->ToTensor();
  const auto &input1 = input[kIndex1]->ToTensor();
  CalBroadCastShape(input0->Shape(), input1->Shape(), &(output->ToTensor()->Shape()));
  output->ToTensor()->SetDtype(ir::DataType::Type::Bool);
  output->ToTensor()->Resize();
  return SUCCESS;
}
OpsErrorCode AclnnBinaryOpBase::CalcWorkspace(const std::vector<const ir::Value *> &input, const ir::Value *output,
                                              size_t *workspaceSize) {
  LOG_OUT << "Begin CalcWorkspace for op [" << name_ << "]";
  executor_->GetWorkspaceSize(static_cast<uint64_t *>(workspaceSize), input[kIndex0]->ToTensor(),
                              input[kIndex1]->ToTensor(), output->ToTensor());
  return SUCCESS;
}

OpsErrorCode AclnnBinaryOpBase::Launch(const std::vector<const ir::Value *> &input, void *workspace,
                                       size_t workspaceSize, ir::Value *output, void *stream) {
  LOG_OUT << "Begin Launch op [" << name_ << "]";
  executor_->Launch(workspace, workspaceSize, stream, input[kIndex0]->ToTensor(), input[kIndex1]->ToTensor(),
                    output->ToTensor());
  return SUCCESS;
}

MRT_REG_OP(eq, AclnnEq, Ascend);
MRT_REG_OP(ne, AclnnNe, Ascend);
MRT_REG_OP(lt, AclnnLt, Ascend);
MRT_REG_OP(le, AclnnLe, Ascend);
MRT_REG_OP(gt, AclnnGt, Ascend);
MRT_REG_OP(ge, AclnnGe, Ascend);
}  // namespace ops
}  // namespace mrt
