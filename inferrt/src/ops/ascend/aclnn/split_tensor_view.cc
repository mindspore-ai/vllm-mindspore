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

#include "ops/ascend/aclnn/split_tensor_view.h"

#include <vector>
#include "ops/ascend/aclnn/utils/opapi_utils.h"
#include "ops/ascend/aclnn/utils/view_utils.h"
#include "ops/op_register.h"

namespace mrt {
namespace ops {
namespace {
void UpdateOutputViewInfo(const ir::TensorPtr &inputTensorPtr, const std::vector<ir::TensorPtr> &outputTensorPtr,
                          const int64_t splitSize, int64_t dim) {
  const auto &curShape = inputTensorPtr->Shape();
  const auto &curStrides = GetTensorStrides(inputTensorPtr);
  auto curOffset = inputTensorPtr->StorageOffset();
  const auto ndim = curShape.size();
  CHECK_IF_FAIL_MSG(ndim > 0, "For SplitTensor, rank should > 0, but got" + std::to_string(ndim));
  const auto wrapDim = DynamicDimWrap(dim, ndim);
  CHECK_IF_FAIL_MSG(splitSize > 0, "For SplitTensor, splitSize must be positive, but got" + std::to_string(splitSize));

  CHECK_IF_FAIL_MSG(!outputTensorPtr.empty(), "For SplitTensor, output tensor size should be greater than 0");
  for (const auto &outputTensor : outputTensorPtr) {
    const auto &inferredShape = outputTensor->Shape();
    CHECK_IF_FAIL_MSG(!outputTensor->HasDynamicShape(),
                      "SplitTensor output shape should have been inferred before CalcWorkspace, but got " +
                        std::to_string(inferredShape.size()) + " dimensions with unresolved values");
    CHECK_IF_FAIL_MSG(inferredShape.size() == ndim, "SplitTensor inferred output rank " +
                                                      std::to_string(inferredShape.size()) +
                                                      " does not match input rank " + std::to_string(ndim));
    CHECK_IF_FAIL_MSG(inferredShape[wrapDim] <= splitSize,
                      "SplitTensor inferred output shape " + ir::ShapeToString(inferredShape) +
                        " has size greater than splitSize " + std::to_string(splitSize) + " at dim " +
                        std::to_string(wrapDim));
    UpdateTensorViewInfo(inputTensorPtr, outputTensor, inferredShape, curStrides, curOffset);
    curOffset += LongToSize(inferredShape[wrapDim] * curStrides[wrapDim]);
  }
}
}  // namespace

OpsErrorCode AclnnSplitTensorView::CalcWorkspace(const std::vector<const ir::Value *> &input, const ir::Value *output,
                                                 size_t *workspaceSize) {
  const auto inputTensorPtr = input[kIndex0]->ToTensor();
  const auto splitSize = input[kIndex1]->ToInt();
  const auto dim = input[kIndex2]->ToInt();
  UpdateOutputViewInfo(inputTensorPtr, output->ToTuple()->ToTensorList(), splitSize, dim);
  CheckStorageMatch(input, output);
  return SUCCESS;
}

MRT_REG_OP(split_tensor_view, AclnnSplitTensorView, Ascend);
}  // namespace ops
}  // namespace mrt
