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

#include "ops/ascend/aclnn/strided_slice_assign.h"
#include "ops/ascend/aclnn/utils/opapi_utils.h"
#include "ops/op_register.h"

namespace mrt {
namespace ops {

constexpr size_t kXIndex = 0;
constexpr size_t kValueIndex = 1;
constexpr size_t kBeginIndex = 2;
constexpr size_t kEndIndex = 3;
constexpr size_t kStridesIndex = 4;
constexpr size_t kAxesIndex = 5;
constexpr int64_t kLastStride = 1;

OpsErrorCode AclnnStridedSliceAssign::CalcWorkspace(const std::vector<const ir::Value *> &input,
                                                    const ir::Value *output, size_t *workspaceSize) {
  x_ = input[kXIndex]->ToTensor();
  value_ = input[kValueIndex]->ToTensor();
  begin_ = input[kBeginIndex]->ToTuple()->ToIntList();
  end_ = input[kEndIndex]->ToTuple()->ToIntList();
  strides_ = input[kStridesIndex]->ToTuple()->ToIntList();
  axes_ = input[kAxesIndex]->IsTuple() ? std::optional(input[kAxesIndex]->ToTuple()->ToIntList()) : std::nullopt;

  // For AclnnStridedSliceAssignV2, the last stride value must be 1.
  if (strides_.back() != kLastStride) {
    auto expandedXShape = x_->Shape();
    auto expandedXStrides = x_->Strides();
    expandedXShape.push_back(kLastStride);
    expandedXStrides.push_back(kLastStride);

    auto expandedValueShape = value_->Shape();
    auto expandedValueStrides = value_->Strides();
    expandedValueShape.push_back(kLastStride);
    expandedValueStrides.push_back(kLastStride);

    x_ = input[kXIndex]->ToTensor()->ShallowClone();
    value_ = input[kValueIndex]->ToTensor()->ShallowClone();
    x_->SetShape(expandedXShape);
    x_->SetStrides(expandedXStrides);
    value_->SetShape(expandedValueShape);
    value_->SetStrides(expandedValueStrides);

    begin_.push_back(0);
    end_.push_back(kLastStride);
    strides_.push_back(kLastStride);
    if (axes_.has_value()) {
      axes_->push_back(expandedXShape.size() - 1);
    }
  }

  executor_->GetWorkspaceSize(static_cast<uint64_t *>(workspaceSize), x_, value_, begin_, end_, strides_, axes_);

  return SUCCESS;
}

OpsErrorCode AclnnStridedSliceAssign::Launch(const std::vector<const ir::Value *> &input, void *workspace,
                                             size_t workspaceSize, ir::Value *output, void *stream) {
  executor_->Launch(workspace, workspaceSize, stream, x_, value_, begin_, end_, strides_, axes_);

  return SUCCESS;
}

MRT_REG_OP(strided_slice_assign, AclnnStridedSliceAssign, Ascend);
}  // namespace ops
}  // namespace mrt
