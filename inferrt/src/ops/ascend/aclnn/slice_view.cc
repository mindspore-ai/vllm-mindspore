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
void UpdateOutputViewInfo(const ir::TensorPtr &input_tensor_ptr, const ir::TensorPtr &output_tensor_ptr,
                          const int64_t ori_dim, const int64_t ori_start, const int64_t ori_end, const int64_t step) {
  const auto &cur_shape = input_tensor_ptr->Shape();
  const auto &cur_strides = GetTensorStrides(input_tensor_ptr);
  const auto dim_size = cur_shape.size();
  CHECK_IF_FAIL_MSG(dim_size > 0, "slice can not be applied to a 0-dim tensor.");
  const auto dim = DynamicDimWrap(ori_dim, dim_size);
  const auto dim_value = cur_shape[dim];

  auto start = ori_start < 0 ? ori_start + dim_value : ori_start;
  if (start < 0) {
    start = 0;
  } else if (start > dim_value) {
    start = dim_value;
  }

  auto end = ori_end < 0 ? ori_end + dim_value : ori_end;
  if (end < start) {
    end = start;
  } else if (end > dim_value) {
    end = dim_value;
  }

  const auto len = end - start;
  auto new_shape = cur_shape;
  new_shape[dim] = (len + step - 1) / step;
  auto new_strides = cur_strides;
  new_strides[dim] *= step;
  const auto storage_offset = input_tensor_ptr->StorageOffset();
  const size_t new_storage_offset = storage_offset + LongToSize(start * cur_strides[dim]);
  UpdateTensorViewInfo(input_tensor_ptr, output_tensor_ptr, new_shape, new_strides, new_storage_offset);
}
}  // namespace

OpsErrorCode AclnnSliceView::CalcWorkspace(const std::vector<const ir::Value *> &input, const ir::Value *output,
                                           size_t *workspaceSize) {
  const auto input_tensor_ptr = input[kIndex0]->ToTensor();
  const auto dim = input[kIndex1]->ToInt();
  const auto start = input[kIndex2]->ToInt();
  const auto end = input[kIndex3]->ToInt();
  const auto step = input[kIndex4]->ToInt();
  CHECK_IF_FAIL_MSG(step > 0, "step must be positive");
  UpdateOutputViewInfo(input_tensor_ptr, output->ToTensor(), dim, start, end, step);
  return SUCCESS;
}

MRT_REG_OP(slice_view, AclnnSliceView, Ascend);
}  // namespace ops
}  // namespace mrt
