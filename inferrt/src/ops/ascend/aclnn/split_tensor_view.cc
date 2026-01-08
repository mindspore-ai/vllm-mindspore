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
  const auto curOffset = inputTensorPtr->StorageOffset();
  const auto ndim = curShape.size();
  CHECK_IF_FAIL_MSG(ndim > 0, "For SplitTensor, rank should > 0, but got" + std::to_string(ndim));
  const auto wrapDim = DynamicDimWrap(dim, ndim);
  CHECK_IF_FAIL_MSG(splitSize > 0, "For SplitTensor, splitSize must be positive, but got" + std::to_string(splitSize));

  const auto numSplits = (curShape[wrapDim] + splitSize - 1) / splitSize;
  if (numSplits <= 0) {
    LOG_EXCEPTION << "For SplitTensor, given input shape: " << curShape << ", splitSize: " << splitSize << ", dim "
                  << dim << ", the output num is 0.";
  }

  for (int64_t idx = 0; idx < numSplits; ++idx) {
    // Calculate the shape and length of sub tensors
    std::vector<int64_t> sliceShape = curShape;
    // Calculate the size of a sub tensor in a specified dimension
    int64_t sliceSize = splitSize;
    if (MS_UNLIKELY(idx == numSplits - 1)) {
      // For the last sub tensor, ensure that it contains all remaining elements in that dimension
      sliceSize = curShape[wrapDim] - (idx * splitSize);
    }
    sliceShape[wrapDim] = sliceSize;
    const size_t newStorageOffset = curOffset + LongToSize(idx * splitSize * curStrides[wrapDim]);
    UpdateTensorViewInfo(inputTensorPtr, outputTensorPtr[idx], sliceShape, curStrides, newStorageOffset);
  }
}
}  // namespace

OpsErrorCode AclnnSplitTensorView::CalcWorkspace(const std::vector<const ir::Value *> &input, const ir::Value *output,
                                                 size_t *workspaceSize) {
  const auto inputTensorPtr = input[kIndex0]->ToTensor();
  const auto splitSize = input[kIndex1]->ToInt();
  const auto dim = input[kIndex2]->ToInt();
  UpdateOutputViewInfo(inputTensorPtr, output->ToTuple()->ToTensorList(), splitSize, dim);
  return SUCCESS;
}

MRT_REG_OP(split_tensor_view, AclnnSplitTensorView, Ascend);
}  // namespace ops
}  // namespace mrt
