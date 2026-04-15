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

#include "ops/ascend/aclnn/permute_view.h"

#include <algorithm>
#include <iterator>
#include <vector>

#include "common/common.h"
#include "ops/ascend/aclnn/utils/view_utils.h"
#include "ops/op_register.h"

namespace mrt {
namespace ops {
namespace {
std::vector<int64_t> NormalizeDims(const std::vector<int64_t> &dims, int64_t dimSize) {
  CHECK_IF_FAIL_MSG(SizeToLong(dims.size()) == dimSize,
                    "permute expects dims length equal to input rank, but got dims size " +
                      std::to_string(dims.size()) + " and input rank " + std::to_string(dimSize));

  std::vector<int64_t> normalized;
  normalized.reserve(dims.size());
  std::transform(dims.begin(), dims.end(), std::back_inserter(normalized),
                 [dimSize](int64_t dim) { return DynamicDimWrap(dim, dimSize); });

  auto sortedDims = normalized;
  std::sort(sortedDims.begin(), sortedDims.end());
  auto uniqueEnd = std::unique(sortedDims.begin(), sortedDims.end());
  CHECK_IF_FAIL_MSG(uniqueEnd == sortedDims.end(), "permute dims must be unique");
  return normalized;
}

void UpdateOutputViewInfo(const ir::TensorPtr &inputTensorPtr, const ir::TensorPtr &outputTensorPtr,
                          const std::vector<int64_t> &dims) {
  const auto &curShape = inputTensorPtr->Shape();
  const auto dimSize = SizeToLong(curShape.size());
  const auto normalizedDims = NormalizeDims(dims, dimSize);
  const auto &curStrides = GetTensorStrides(inputTensorPtr);

  std::vector<int64_t> newShape;
  std::vector<int64_t> newStrides;
  newShape.reserve(curShape.size());
  newStrides.reserve(curStrides.size());
  for (auto dim : normalizedDims) {
    newShape.emplace_back(curShape[LongToSize(dim)]);
    newStrides.emplace_back(curStrides[LongToSize(dim)]);
  }

  UpdateTensorViewInfo(inputTensorPtr, outputTensorPtr, newShape, newStrides, inputTensorPtr->StorageOffset());
}
}  // namespace

OpsErrorCode AclnnPermuteView::CalcWorkspace(const std::vector<const ir::Value *> &input, const ir::Value *output,
                                             size_t *workspaceSize) {
  const auto inputTensorPtr = input[kIndex0]->ToTensor();
  const auto &dims = input[kIndex1]->ToTuple()->ToIntList();
  UpdateOutputViewInfo(inputTensorPtr, output->ToTensor(), dims);
  CheckStorageMatch(input, output);
  *workspaceSize = 0;
  return SUCCESS;
}

MRT_REG_OP(permute_view, AclnnPermuteView, Ascend);
}  // namespace ops
}  // namespace mrt
