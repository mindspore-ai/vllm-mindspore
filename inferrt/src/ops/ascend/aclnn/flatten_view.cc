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
bool IsFlattenRangeContiguous(const std::vector<int64_t> &curShape, const std::vector<int64_t> &curStrides,
                              int64_t startDim, int64_t endDim) {
  // The dims in [startDim, endDim] can be flattened as a view iff they are
  // contiguous in memory: for each i in [startDim, endDim),
  // strides[i] == shape[i+1] * strides[i+1] (for non-1-sized dims).
  // Walk from endDim back to startDim, skipping size-1 dims.
  for (int64_t i = endDim; i > startDim; --i) {
    if (curShape[i] == 1) {
      continue;
    }
    int64_t prev = i - 1;
    while (prev > startDim && curShape[prev] == 1) {
      --prev;
    }
    if (curStrides[prev] != curShape[i] * curStrides[i]) {
      return false;
    }
  }
  return true;
}

void UpdateOutputViewInfo(const ir::TensorPtr &inputTensorPtr, const ir::TensorPtr &outputTensorPtr, int64_t startDim,
                          int64_t endDim) {
  const auto &curShape = inputTensorPtr->Shape();
  const auto &curStrides = GetTensorStrides(inputTensorPtr);
  auto dimSize = static_cast<int64_t>(curShape.size());

  // Normalize negative dims
  startDim = DynamicDimWrap(startDim, dimSize);
  endDim = DynamicDimWrap(endDim, dimSize);

  if (!IsFlattenRangeContiguous(curShape, curStrides, startDim, endDim)) {
    LOG_EXCEPTION << "Flatten view requires dims [" << startDim << ", " << endDim
                  << "] to be contiguous in memory, but got shape " << curShape << " and strides " << curStrides;
  }

  // Compute new shape and strides:
  // dims before startDim stay, dims from startDim to endDim are flattened into one,
  // dims after endDim stay
  std::vector<int64_t> newShape;
  std::vector<int64_t> newStrides;

  // Dims before startDim
  for (int64_t i = 0; i < startDim; ++i) {
    newShape.emplace_back(curShape[i]);
    newStrides.emplace_back(curStrides[i]);
  }

  // Flattened dim: product of sizes from startDim to endDim, stride is curStrides[endDim]
  int64_t flatSize = 1;
  for (int64_t i = startDim; i <= endDim; ++i) {
    flatSize *= curShape[i];
  }
  newShape.emplace_back(flatSize);
  newStrides.emplace_back(curStrides[endDim]);

  // Dims after endDim
  for (int64_t i = endDim + 1; i < dimSize; ++i) {
    newShape.emplace_back(curShape[i]);
    newStrides.emplace_back(curStrides[i]);
  }

  UpdateTensorViewInfo(inputTensorPtr, outputTensorPtr, newShape, newStrides, inputTensorPtr->StorageOffset());
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
