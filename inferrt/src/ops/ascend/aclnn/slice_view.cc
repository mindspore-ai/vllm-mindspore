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

#include "ops/ascend/aclnn/slice_view.h"

#include <vector>
#include "ops/ascend/aclnn/utils/opapi_utils.h"
#include "ops/ascend/aclnn/utils/view_utils.h"
#include "ops/op_register.h"

namespace mrt {
namespace ops {
namespace {
void UpdateOutputViewInfo(const ir::TensorPtr &inputTensorPtr, const ir::TensorPtr &outputTensorPtr,
                          const int64_t oriDim, const int64_t oriStart, const int64_t oriEnd, const int64_t step) {
  const auto &curShape = inputTensorPtr->Shape();
  const auto &curStrides = GetTensorStrides(inputTensorPtr);
  const auto dimSize = curShape.size();
  CHECK_IF_FAIL_MSG(dimSize > 0, "slice can not be applied to a 0-dim tensor.");
  const auto dim = DynamicDimWrap(oriDim, dimSize);
  const auto dimValue = curShape[dim];

  auto start = oriStart < 0 ? oriStart + dimValue : oriStart;
  if (start < 0) {
    start = 0;
  } else if (start > dimValue) {
    start = dimValue;
  }

  auto end = oriEnd < 0 ? oriEnd + dimValue : oriEnd;
  if (end < start) {
    end = start;
  } else if (end > dimValue) {
    end = dimValue;
  }

  const auto len = end - start;
  auto newShape = curShape;
  newShape[dim] = (len + step - 1) / step;
  auto newStrides = curStrides;
  newStrides[dim] *= step;
  const auto storageOffset = inputTensorPtr->StorageOffset();
  const size_t newStorageOffset = storageOffset + LongToSize(start * curStrides[dim]);
  UpdateTensorViewInfo(inputTensorPtr, outputTensorPtr, newShape, newStrides, newStorageOffset);
}
}  // namespace

OpsErrorCode AclnnSliceView::CalcWorkspace(const std::vector<const ir::Value *> &input, const ir::Value *output,
                                           size_t *workspaceSize) {
  const auto inputTensorPtr = input[kIndex0]->ToTensor();
  const auto dim = input[kIndex1]->ToInt();
  const auto start = input[kIndex2]->ToInt();
  const auto end = input[kIndex3]->ToInt();
  const auto step = input[kIndex4]->ToInt();
  CHECK_IF_FAIL_MSG(step > 0, "step must be positive");
  UpdateOutputViewInfo(inputTensorPtr, output->ToTensor(), dim, start, end, step);
  CheckStorageMatch(input, output);
  return SUCCESS;
}

MRT_REG_OP(slice_view, AclnnSliceView, Ascend);
}  // namespace ops
}  // namespace mrt
