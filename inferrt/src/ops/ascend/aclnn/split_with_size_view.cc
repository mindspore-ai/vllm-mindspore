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

#include "ops/ascend/aclnn/split_with_size_view.h"

#include <vector>
#include "ops/ascend/aclnn/utils/opapi_utils.h"
#include "ops/ascend/aclnn/utils/view_utils.h"
#include "ops/op_register.h"

namespace mrt {
namespace ops {
namespace {
void SplitSizeInputsCheck(const std::vector<int64_t> &split_size, const int64_t &axis,
                          const std::vector<int64_t> &tensor_shape) {
  CHECK_IF_FAIL_MSG(split_size.size() > 0, "For SplitWithSize, the size of split_size should > 0, but got" +
                                             std::to_string(split_size.size()));
  const int64_t sum_split_size = std::accumulate(split_size.begin(), split_size.end(), 0);
  if (sum_split_size != tensor_shape[axis]) {
    LOG_EXCEPTION << "For 'SplitWithSize',  the sum of split_size should be equal to " << tensor_shape[axis]
                  << "(input.shape[" << axis << "]), but got split_sizes: " << split_size;
  }
}

void UpdateOutputViewInfo(const ir::TensorPtr &input_tensor_ptr, const std::vector<ir::TensorPtr> &output_tensor_vector,
                          const std::vector<int64_t> &split_size, int64_t dim) {
  const auto &cur_shape = input_tensor_ptr->Shape();
  const auto &cur_strides = GetTensorStrides(input_tensor_ptr);
  auto cur_offset = input_tensor_ptr->StorageOffset();
  const auto rank = SizeToLong(cur_shape.size());
  CHECK_IF_FAIL_MSG(rank > 0, "For SplitWithSize, rank should > 0, but got" + std::to_string(rank));
  const auto ndim = cur_shape.size();
  const auto wrap_dim = DynamicDimWrap(dim, ndim);
  SplitSizeInputsCheck(split_size, wrap_dim, cur_shape);

  CHECK_IF_FAIL_MSG(split_size.size() == output_tensor_vector.size(),
                    "For SplitWithSize, the size of split_size is " + std::to_string(split_size.size()) +
                      " and the size of output_tensor_vector is " + std::to_string(output_tensor_vector.size()));
  for (size_t i = 0; i < split_size.size(); ++i) {
    const auto split_iter = split_size[i];
    std::vector<int64_t> slice_shape(cur_shape);
    slice_shape[wrap_dim] = split_iter;
    UpdateTensorViewInfo(input_tensor_ptr, output_tensor_vector[i], slice_shape, cur_strides, cur_offset);
    cur_offset += LongToSize(split_iter * cur_strides[wrap_dim]);
  }
}
}  // namespace

OpsErrorCode AclnnSplitWithSizeView::CalcWorkspace(const std::vector<const ir::Value *> &input, const ir::Value *output,
                                                   size_t *workspaceSize) {
  const auto input_tensor_ptr = input[kIndex0]->ToTensor();
  const auto split_size = input[kIndex1]->ToTuple()->ToIntList();
  const auto dim = input[kIndex2]->ToInt();
  UpdateOutputViewInfo(input_tensor_ptr, output->ToTuple()->ToTensorList(), split_size, dim);
  return SUCCESS;
}

MRT_REG_OP(split_with_size_view, AclnnSplitWithSizeView, Ascend);
}  // namespace ops
}  // namespace mrt
