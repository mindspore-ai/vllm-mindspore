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
void UpdateOutputViewInfo(const ir::TensorPtr &input_tensor_ptr, const std::vector<ir::TensorPtr> &output_tensor_vector,
                          const int64_t split_size, int64_t dim) {
  const auto &cur_shape = input_tensor_ptr->Shape();
  const auto &cur_strides = GetTensorStrides(input_tensor_ptr);
  const auto cur_offset = input_tensor_ptr->StorageOffset();
  const auto ndim = cur_shape.size();
  CHECK_IF_FAIL_MSG(ndim > 0, "For SplitTensor, rank should > 0, but got" + std::to_string(ndim));
  const auto wrap_dim = DynamicDimWrap(dim, ndim);
  CHECK_IF_FAIL_MSG(split_size > 0,
                    "For SplitTensor, split_size must be positive, but got" + std::to_string(split_size));

  const auto num_splits = (cur_shape[wrap_dim] + split_size - 1) / split_size;
  if (num_splits <= 0) {
    LOG_EXCEPTION << "For SplitTensor, given input shape: " << cur_shape << ", split_size: " << split_size << ", dim "
                  << dim << ", the output num is 0.";
  }

  for (int64_t idx = 0; idx < num_splits; ++idx) {
    // Calculate the shape and length of sub tensors
    std::vector<int64_t> slice_shape = cur_shape;
    // Calculate the size of a sub tensor in a specified dimension
    int64_t slice_size = split_size;
    if (MS_UNLIKELY(idx == num_splits - 1)) {
      // For the last sub tensor, ensure that it contains all remaining elements in that dimension
      slice_size = cur_shape[wrap_dim] - (idx * split_size);
    }
    slice_shape[wrap_dim] = slice_size;
    const size_t new_storage_offset = cur_offset + LongToSize(idx * split_size * cur_strides[wrap_dim]);
    UpdateTensorViewInfo(input_tensor_ptr, output_tensor_vector[idx], slice_shape, cur_strides, new_storage_offset);
  }
}
}  // namespace

OpsErrorCode AclnnSplitTensorView::CalcWorkspace(const std::vector<const ir::Value *> &input, const ir::Value *output,
                                                 size_t *workspaceSize) {
  const auto input_tensor_ptr = input[kIndex0]->ToTensor();
  const auto split_size = input[kIndex1]->ToInt();
  const auto dim = input[kIndex2]->ToInt();
  UpdateOutputViewInfo(input_tensor_ptr, output->ToTuple()->ToTensorList(), split_size, dim);
  return SUCCESS;
}

MRT_REG_OP(split_tensor_view, AclnnSplitTensorView, Ascend);
}  // namespace ops
}  // namespace mrt
