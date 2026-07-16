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

#include "ops/ascend/aclnn/unflatten_view.h"

#include <algorithm>
#include <vector>

#include "common/common.h"
#include "ops/ascend/aclnn/utils/view_utils.h"
#include "ops/op_register.h"

namespace mrt {
namespace ops {
namespace {
void UpdateOutputViewInfo(const ir::TensorPtr &inputTensorPtr, const ir::TensorPtr &outputTensorPtr, int64_t dim,
                          const std::vector<int64_t> &sizes) {
  if (sizes.empty()) {
    RT_GLOG(EXCEPTION) << "unflatten: sizes must be non-empty";
  }
  if (std::any_of(sizes.begin(), sizes.end(), [](int64_t size) { return size < -1; })) {
    RT_GLOG(EXCEPTION) << "unflatten: sizes must not contain values less than -1";
  }

  const auto &curShape = inputTensorPtr->Shape();
  const auto &curStrides = GetTensorStrides(inputTensorPtr);
  const auto &inferredShape = outputTensorPtr->Shape();
  CHECK_IF_FAIL_MSG(!outputTensorPtr->HasDynamicShape(),
                    "Unflatten output shape should have been inferred before CalcWorkspace, but got " +
                      std::to_string(inferredShape.size()) + " dimensions with unresolved values");
  dim = DynamicDimWrap(dim, SizeToLong(curShape.size()));

  const auto newStrides = CalculateViewStrides(curShape, curStrides, inferredShape);
  if (!newStrides.has_value()) {
    RT_GLOG(EXCEPTION)
      << "Unflatten encountered unsupported non-contiguous input tensor. output shape: " << inferredShape
      << ", unflatten dim: " << dim << ", unflatten sizes: " << sizes << ", input shape: " << curShape
      << ", input stride: " << curStrides
      << ". Consider calling .contiguous() on the input tensor at the corresponding operator call site.";
  }
  UpdateTensorViewInfo(inputTensorPtr, outputTensorPtr, inferredShape, newStrides.value(),
                       inputTensorPtr->StorageOffset());
}
}  // namespace

OpsErrorCode AclnnUnflattenView::CalcWorkspace(const std::vector<const ir::Value *> &input, const ir::Value *output,
                                               size_t *workspaceSize) {
  const auto inputTensorPtr = input[kIndex0]->ToTensor();
  const auto dim = input[kIndex1]->ToInt();
  const auto &sizes = input[kIndex2]->ToTuple()->ToIntList();
  UpdateOutputViewInfo(inputTensorPtr, output->ToTensor(), dim, sizes);
  CheckStorageMatch(input, output);
  *workspaceSize = 0;
  return SUCCESS;
}

MRT_REG_OP(unflatten_view, AclnnUnflattenView, Ascend);
}  // namespace ops
}  // namespace mrt
