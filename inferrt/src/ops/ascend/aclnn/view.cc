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

#include "ops/ascend/aclnn/view.h"

#include <vector>
#include "ops/ascend/aclnn/utils/opapi_utils.h"
#include "ops/ascend/aclnn/utils/view_utils.h"
#include "ops/op_register.h"

namespace mrt {
namespace ops {
namespace {
void UpdateOutputViewInfo(const ir::TensorPtr &inputTensorPtr, const ir::TensorPtr &outputTensorPtr,
                          const std::vector<int64_t> &viewShapeArg) {
  const auto &curShape = inputTensorPtr->Shape();
  const auto &curStrides = GetTensorStrides(inputTensorPtr);
  const auto &inferredShape = outputTensorPtr->Shape();
  CHECK_IF_FAIL_MSG(!outputTensorPtr->HasDynamicShape(),
                    "View output shape should have been inferred before CalcWorkspace, but got " +
                      std::to_string(inferredShape.size()) + " dimensions with unresolved values");

  const auto strides = CalculateViewStrides(curShape, curStrides, inferredShape);
  if (strides.has_value()) {
    UpdateTensorViewInfo(inputTensorPtr, outputTensorPtr, inferredShape, strides.value());
    return;
  }
  LOG_EXCEPTION << "View encountered unsupported non-contiguous input tensor. output shape: " << viewShapeArg
                << " (inferred as " << inferredShape << "), input shape: " << curShape
                << ", input stride: " << curStrides
                << ". Consider calling .contiguous() on the input tensor at the corresponding operator call site.";
}
}  // namespace

OpsErrorCode AclnnView::CalcWorkspace(const std::vector<const ir::Value *> &input, const ir::Value *output,
                                      size_t *workspaceSize) {
  const auto inputTensorPtr = input[kIndex0]->ToTensor();
  const auto &shape = input[kIndex1]->ToTuple()->ToIntList();
  if (std::any_of(shape.begin(), shape.end(), [](const int &shapeI) { return shapeI < -1; })) {
    LOG_EXCEPTION << "For View the component of shape can't be less than -1";
  }
  UpdateOutputViewInfo(inputTensorPtr, output->ToTensor(), shape);
  CheckStorageMatch(input, output);
  return SUCCESS;
}

MRT_REG_OP(view, AclnnView, Ascend);
}  // namespace ops
}  // namespace mrt
