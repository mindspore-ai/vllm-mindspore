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

#include "ops/ascend/aclnn/select_view.h"

#include <vector>
#include "ops/ascend/aclnn/utils/opapi_utils.h"
#include "ops/ascend/aclnn/utils/view_utils.h"
#include "ops/op_register.h"

namespace mrt {
namespace ops {
namespace {
void UpdateOutputViewInfo(const ir::TensorPtr &input_tensor_ptr, const ir::TensorPtr &output_tensor_ptr,
                          const int64_t ori_dim, const int64_t ori_index) {
  const auto &cur_shape = input_tensor_ptr->Shape();
  const auto dim_size = cur_shape.size();
  CHECK_IF_FAIL_MSG(dim_size > 0, "For Primitive [SelectExtView] rank must >= 1");
  const auto dim = DynamicDimWrap(ori_dim, dim_size);
  const auto dim_value = cur_shape[dim];
  CHECK_IF_FAIL_MSG(ori_index >= -dim_value && ori_index < dim_value,
                    "For Primitive [SelectExtView] start exceed range. start: " + std::to_string(ori_index) +
                      ", start should be in [" + std::to_string(-dim_value) + ", " + std::to_string(dim_value) + ").");

  const auto index = ori_index < 0 ? ori_index + dim_value : ori_index;
  auto new_shape = cur_shape;
  const auto &cur_strides = GetTensorStrides(input_tensor_ptr);
  auto new_strides = cur_strides;
  const auto storage_offset = input_tensor_ptr->StorageOffset();
  const size_t new_storage_offset = storage_offset + LongToSize(index * cur_strides[dim]);
  new_shape.erase(new_shape.begin() + dim);
  new_strides.erase(new_strides.begin() + dim);
  UpdateTensorViewInfo(input_tensor_ptr, output_tensor_ptr, new_shape, new_strides, new_storage_offset);
}
}  // namespace

OpsErrorCode AclnnSelectView::CalcWorkspace(const std::vector<const ir::Value *> &input, const ir::Value *output,
                                            size_t *workspaceSize) {
  const auto input_tensor_ptr = input[kIndex0]->ToTensor();
  const auto dim = input[kIndex1]->ToInt();
  const auto index = input[kIndex2]->ToInt();
  UpdateOutputViewInfo(input_tensor_ptr, output->ToTensor(), dim, index);
  return SUCCESS;
}

MRT_REG_OP(select_view, AclnnSelectView, Ascend);
}  // namespace ops
}  // namespace mrt
