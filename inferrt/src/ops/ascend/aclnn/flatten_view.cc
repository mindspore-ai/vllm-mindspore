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
#include <optional>
#include <vector>

#include "ops/ascend/aclnn/utils/view_utils.h"
#include "ops/op_register.h"

namespace mrt {
namespace ops {
namespace {
std::optional<std::vector<int64_t>> CalculateViewStrides(const std::vector<int64_t> &curShape,
                                                         const std::vector<int64_t> &curStrides,
                                                         const std::vector<int64_t> &newShape) {
  if (curShape.empty()) {
    return std::vector<int64_t>(newShape.size(), 1);
  }

  bool isOldEmpty = std::any_of(curShape.begin(), curShape.end(), [](int64_t dim) { return dim == 0; });
  if (isOldEmpty && curShape == newShape) {
    return curStrides;
  }

  const int64_t newRank = SizeToLong(newShape.size());
  std::vector<int64_t> newStrides(newRank, 0);
  if (isOldEmpty) {
    for (int64_t dim = newRank - 1; dim >= 0; --dim) {
      if (dim == (newRank - 1)) {
        newStrides[dim] = 1;
      } else {
        newStrides[dim] =
          std::max(newShape[LongToSize(dim + 1)], static_cast<int64_t>(1)) * newStrides[LongToSize(dim + 1)];
      }
    }
    return newStrides;
  }

  int64_t viewDim = newRank - 1;
  int64_t baseStride = curStrides.back();
  int64_t tensorElems = 1;
  int64_t viewElems = 1;
  for (int64_t dim = SizeToLong(curShape.size()) - 1; dim >= 0; --dim) {
    tensorElems *= curShape[LongToSize(dim)];
    if (dim == 0 ||
        (curShape[LongToSize(dim - 1)] != 1 && curStrides[LongToSize(dim - 1)] != tensorElems * baseStride)) {
      while (viewDim >= 0 && (viewElems < tensorElems || newShape[LongToSize(viewDim)] == 1)) {
        newStrides[LongToSize(viewDim)] = viewElems * baseStride;
        viewElems *= newShape[LongToSize(viewDim)];
        --viewDim;
      }
      if (viewElems != tensorElems) {
        return std::nullopt;
      }
      if (dim > 0) {
        baseStride = curStrides[LongToSize(dim - 1)];
        tensorElems = 1;
        viewElems = 1;
      }
    }
  }
  if (viewDim != -1) {
    return std::nullopt;
  }
  return newStrides;
}

void UpdateOutputViewInfo(const ir::TensorPtr &inputTensorPtr, const ir::TensorPtr &outputTensorPtr, int64_t startDim,
                          int64_t endDim) {
  const auto &curShape = inputTensorPtr->Shape();
  const auto &curStrides = GetTensorStrides(inputTensorPtr);
  auto dimSize = static_cast<int64_t>(curShape.size());

  if (dimSize == 0) {
    UpdateTensorViewInfo(inputTensorPtr, outputTensorPtr, {1}, {1}, inputTensorPtr->StorageOffset());
    return;
  }

  // Normalize negative dims
  startDim = DynamicDimWrap(startDim, dimSize);
  endDim = DynamicDimWrap(endDim, dimSize);
  if (startDim > endDim) {
    LOG_EXCEPTION << "flatten() has invalid args: start_dim cannot come after end_dim";
  }

  // Build flatten target shape.
  std::vector<int64_t> newShape;
  newShape.reserve(curShape.size() - static_cast<size_t>(endDim - startDim));

  for (int64_t i = 0; i < startDim; ++i) {
    newShape.emplace_back(curShape[LongToSize(i)]);
  }

  int64_t flatSize = 1;
  for (int64_t i = startDim; i <= endDim; ++i) {
    flatSize *= curShape[LongToSize(i)];
  }
  newShape.emplace_back(flatSize);

  for (int64_t i = endDim + 1; i < dimSize; ++i) {
    newShape.emplace_back(curShape[LongToSize(i)]);
  }

  const auto newStrides = CalculateViewStrides(curShape, curStrides, newShape);
  if (!newStrides.has_value()) {
    LOG_EXCEPTION << "Flatten view shape " << newShape << " is not compatible with input tensor's shape " << curShape
                  << " and stride " << curStrides;
  }
  UpdateTensorViewInfo(inputTensorPtr, outputTensorPtr, newShape, newStrides.value(), inputTensorPtr->StorageOffset());
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
