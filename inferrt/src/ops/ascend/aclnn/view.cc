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
std::optional<std::vector<int64_t>> CalculateViewStrides(const std::vector<int64_t> &curShape,
                                                         const std::vector<int64_t> &curStrides,
                                                         const std::vector<int64_t> &newShape) {
  if (curShape.empty()) {
    return std::vector<int64_t>(newShape.size(), 1);
  }

  bool isOldEmpty = std::any_of(curShape.begin(), curShape.end(), [](const int64_t dim) { return dim == 0; });
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
        newStrides[dim] = std::max(newShape[dim + 1], static_cast<int64_t>(1)) * newStrides[dim + 1];
      }
    }
    return newStrides;
  }

  int64_t viewDim = newRank - 1;
  int64_t baseStride = curStrides.back();
  int64_t tensorElems = 1;
  int64_t viewElems = 1;
  for (int64_t dim = SizeToLong(curShape.size()) - 1; dim >= 0; --dim) {
    tensorElems *= curShape[dim];
    if (dim == 0 || (curShape[dim - 1] != 1 && curStrides[dim - 1] != tensorElems * baseStride)) {
      while (viewDim >= 0 && (viewElems < tensorElems || newShape[viewDim] == 1)) {
        newStrides[viewDim] = viewElems * baseStride;
        viewElems *= newShape[viewDim];
        --viewDim;
      }
      if (viewElems != tensorElems) {
        return std::nullopt;
      }
      if (dim > 0) {
        baseStride = curStrides[dim - 1];
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

std::vector<int64_t> InferSizeImpl(const std::vector<int64_t> &newShape, int64_t numElements) {
  int64_t newSize = 1;
  std::optional<int64_t> inferDim;
  for (int64_t dim = 0, ndim = static_cast<int64_t>(newShape.size()); dim != ndim; ++dim) {
    if (newShape[dim] == -1) {
      if (inferDim) {
        LOG_EXCEPTION << "only one dimension can be inferred";
      }
      inferDim = dim;
    } else if (newShape[dim] >= 0) {
      newSize *= newShape[dim];
    } else {
      LOG_EXCEPTION << "invalid proposed_shape dimension";
    }
  }

  if (numElements == newSize || (inferDim && newSize > 0 && numElements % newSize == 0)) {
    std::vector<int64_t> res(newShape);
    if (inferDim) {
      if (newSize == 0) {
        LOG_OUT << "WARNING: cannot reshape tensor of 0 elements into proposed_shape, because the unspecified "
                   "dimension size -1 can be any value and is ambiguous";
        res[*inferDim] = 0;
      } else {
        res[*inferDim] = numElements / newSize;
      }
    }
    return res;
  }
  LOG_EXCEPTION << "proposed_shape is invalid for input of size";
  return {};
}

std::vector<int64_t> InferShape(const std::vector<int64_t> &newShape, const std::vector<int64_t> &curShape) {
  const int64_t numElements =
    std::accumulate(curShape.begin(), curShape.end(), static_cast<int64_t>(1), std::multiplies<int64_t>());
  auto res = InferSizeImpl(newShape, numElements);
  return res;
}

void UpdateOutputViewInfo(const ir::TensorPtr &inputTensorPtr, const ir::TensorPtr &outputTensorPtr,
                          const std::vector<int64_t> &newShape) {
  const auto &curShape = inputTensorPtr->Shape();
  const auto &curStrides = GetTensorStrides(inputTensorPtr);
  const auto inferShape = InferShape(newShape, curShape);
  const auto strides = CalculateViewStrides(curShape, curStrides, inferShape);
  if (strides.has_value()) {
    UpdateTensorViewInfo(inputTensorPtr, outputTensorPtr, inferShape, strides.value());
    return;
  }
  LOG_EXCEPTION << "View shape " << newShape << "is not compatible with input tensor's shape " << curShape
                << " and stride " << curStrides;
}
}  // namespace

OpsErrorCode AclnnView::CalcWorkspace(const std::vector<const ir::Value *> &input, const ir::Value *output,
                                      size_t *workspaceSize) {
  const auto inputTensorPtr = input[kIndex0]->ToTensor();
  if (!inputTensorPtr->IsContiguous()) {
    LOG_EXCEPTION << "Input tensor is not contiguous";
  }
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
