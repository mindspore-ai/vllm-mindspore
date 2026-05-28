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

#include "ops/ascend/aclnn/flatten_view.h"

#include <numeric>
#include <vector>

#include "ops/ascend/aclnn/utils/view_utils.h"
#include "ops/op_register.h"

namespace mrt {
namespace ops {
namespace {
void UpdateOutputViewInfo(const ir::TensorPtr &inputTensorPtr, const ir::TensorPtr &outputTensorPtr, int64_t startDim,
                          int64_t endDim) {
  const auto &curShape = inputTensorPtr->Shape();
  const auto &curStrides = GetTensorStrides(inputTensorPtr);
  const auto &inferredShape = outputTensorPtr->Shape();
  CHECK_IF_FAIL_MSG(!outputTensorPtr->HasDynamicShape(),
                    "Flatten output shape should have been inferred before CalcWorkspace, but got " +
                      std::to_string(inferredShape.size()) + " dimensions with unresolved values");
  auto dimSize = static_cast<int64_t>(curShape.size());

  if (dimSize == 0) {
    UpdateTensorViewInfo(inputTensorPtr, outputTensorPtr, inferredShape, std::vector<int64_t>(inferredShape.size(), 1),
                         inputTensorPtr->StorageOffset());
    return;
  }

  startDim = DynamicDimWrap(startDim, dimSize);
  endDim = DynamicDimWrap(endDim, dimSize);
  if (startDim > endDim) {
    LOG_EXCEPTION << "flatten() has invalid args: start_dim cannot come after end_dim";
  }

  const auto newStrides = CalculateViewStrides(curShape, curStrides, inferredShape);
  if (!newStrides.has_value()) {
    LOG_EXCEPTION << "Flatten output shape " << inferredShape << " from flatten(" << startDim << ", " << endDim
                  << ") is not compatible with input tensor's shape " << curShape << " and stride " << curStrides;
  }
  UpdateTensorViewInfo(inputTensorPtr, outputTensorPtr, inferredShape, newStrides.value(),
                       inputTensorPtr->StorageOffset());
}
}  // namespace

OpsErrorCode AclnnFlattenView::CalcWorkspace(const std::vector<const ir::Value *> &input, const ir::Value *output,
                                             size_t *workspaceSize) {
  const auto inputTensorPtr = input[kIndex0]->ToTensor();
  const auto startDim = input[kIndex1]->ToInt();
  const auto endDim = input[kIndex2]->ToInt();
  UpdateOutputViewInfo(inputTensorPtr, output->ToTensor(), startDim, endDim);
  CheckStorageMatch(input, output);
  *workspaceSize = 0;
  return SUCCESS;
}

MRT_REG_OP(flatten, AclnnFlattenView, Ascend);
}  // namespace ops
}  // namespace mrt
