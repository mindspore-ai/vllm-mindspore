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

#include "ops/ascend/aclnn/select_view.h"

#include <vector>
#include "ops/ascend/aclnn/utils/opapi_utils.h"
#include "ops/ascend/aclnn/utils/view_utils.h"
#include "ops/op_register.h"

namespace mrt {
namespace ops {
namespace {
void UpdateOutputViewInfo(const ir::TensorPtr &inputTensorPtr, const ir::TensorPtr &outputTensorPtr,
                          const int64_t oriDim, const int64_t oriIndex) {
  const auto &curShape = inputTensorPtr->Shape();
  const auto dimSize = curShape.size();
  CHECK_IF_FAIL_MSG(dimSize > 0, "For Primitive [SelectExtView] rank must >= 1");
  const auto dim = DynamicDimWrap(oriDim, dimSize);
  const auto dimValue = curShape[dim];
  CHECK_IF_FAIL_MSG(oriIndex >= -dimValue && oriIndex < dimValue,
                    "For Primitive [SelectExtView] start exceed range. start: " + std::to_string(oriIndex) +
                      ", start should be in [" + std::to_string(-dimValue) + ", " + std::to_string(dimValue) + ").");

  const auto index = oriIndex < 0 ? oriIndex + dimValue : oriIndex;
  auto newShape = curShape;
  const auto &curStrides = GetTensorStrides(inputTensorPtr);
  auto newStrides = curStrides;
  const auto storageOffset = inputTensorPtr->StorageOffset();
  const size_t newStorageOffset = storageOffset + LongToSize(index * curStrides[dim]);
  newShape.erase(newShape.begin() + dim);
  newStrides.erase(newStrides.begin() + dim);
  UpdateTensorViewInfo(inputTensorPtr, outputTensorPtr, newShape, newStrides, newStorageOffset);
}
}  // namespace

OpsErrorCode AclnnSelectView::CalcWorkspace(const std::vector<const ir::Value *> &input, const ir::Value *output,
                                            size_t *workspaceSize) {
  const auto inputTensorPtr = input[kIndex0]->ToTensor();
  const auto dim = input[kIndex1]->ToInt();
  const auto index = input[kIndex2]->ToInt();
  UpdateOutputViewInfo(inputTensorPtr, output->ToTensor(), dim, index);
  return SUCCESS;
}

MRT_REG_OP(select_view, AclnnSelectView, Ascend);
}  // namespace ops
}  // namespace mrt
