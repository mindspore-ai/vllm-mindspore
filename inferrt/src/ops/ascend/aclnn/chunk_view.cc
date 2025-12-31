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

#include "ops/ascend/aclnn/chunk_view.h"

#include <vector>
#include "ops/ascend/aclnn/utils/opapi_utils.h"
#include "ops/ascend/aclnn/utils/view_utils.h"
#include "ops/op_register.h"

namespace mrt {
namespace ops {
namespace {
void UpdateOutputViewInfo(const ir::TensorPtr &inputTensorPtr, const std::vector<ir::TensorPtr> &outputTensorVector,
                          int64_t chunks, int64_t dim) {
  const auto &curShape = inputTensorPtr->Shape();
  const auto &curStrides = GetTensorStrides(inputTensorPtr);
  const auto curOffset = inputTensorPtr->StorageOffset();
  const auto ndim = curShape.size();
  CHECK_IF_FAIL_MSG(ndim > 0, "For 'Chunk', input's rank should be greater than 0, but got " + std::to_string(ndim));
  CHECK_IF_FAIL_MSG(chunks > 0, "For 'Chunk', chunks should be greater than 0, but got " + std::to_string(chunks));

  const auto wrapDim = DynamicDimWrap(dim, ndim);
  const int64_t dimSize = curShape[wrapDim];
  const int64_t splitSize = (dimSize + chunks - 1) / chunks;
  if (MS_UNLIKELY(dimSize == 0)) {
    if (splitSize == 0) {
      CHECK_IF_FAIL_MSG(static_cast<int64_t>(outputTensorVector.size()) == chunks,
                        "For 'Chunk', output tensor size (" + std::to_string(outputTensorVector.size()) +
                          ") does not match expected chunks (" + std::to_string(chunks) + ")");
      for (int64_t i = 0; i < chunks; ++i) {
        UpdateTensorViewInfo(inputTensorPtr, outputTensorVector[i], curShape, curStrides);
      }
      return;
    }
    LOG_EXCEPTION << "For 'Chunk', output_num must be positive, but got 0";
  }

  // Calculate the number of sub tensors after segmentation
  const auto numSplits = std::max<int64_t>((dimSize + splitSize - 1) / splitSize, 1);
  const auto lastSplitSize = splitSize - (splitSize * numSplits - dimSize);
  CHECK_IF_FAIL_MSG(static_cast<int64_t>(outputTensorVector.size()) == numSplits,
                    "For 'Chunk', output tensor size (" + std::to_string(outputTensorVector.size()) +
                      ") does not match expected chunks (" + std::to_string(chunks) + ")");
  for (int64_t idx = 0; idx < numSplits; ++idx) {
    // Calculate the shape and length of sub tensors
    std::vector<int64_t> sliceShape = curShape;

    // Calculate the size of a sub tensor in a specified dimension
    sliceShape[wrapDim] = (idx == numSplits - 1) ? lastSplitSize : splitSize;
    // Calculate the storage offset of sub tensors
    const size_t newStorageOffset = curOffset + LongToSize(idx * splitSize * curStrides[wrapDim]);
    UpdateTensorViewInfo(inputTensorPtr, outputTensorVector[idx], sliceShape, curStrides, newStorageOffset);
  }
}
}  // namespace

OpsErrorCode AclnnChunkView::CalcWorkspace(const std::vector<const ir::Value *> &input, const ir::Value *output,
                                           size_t *workspaceSize) {
  const auto inputTensorPtr = input[kIndex0]->ToTensor();
  const auto chunks = input[kIndex1]->ToInt();
  CHECK_IF_FAIL_MSG(chunks >= 0, "chunks must be positive, but got " + std::to_string(chunks));
  const auto dim = input[kIndex2]->ToInt();
  UpdateOutputViewInfo(inputTensorPtr, output->ToTuple()->ToTensorList(), chunks, dim);
  return SUCCESS;
}

MRT_REG_OP(chunk_view, AclnnChunkView, Ascend);
}  // namespace ops
}  // namespace mrt
