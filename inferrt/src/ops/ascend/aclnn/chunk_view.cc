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
void UpdateOutputViewInfo(const ir::TensorPtr &input_tensor_ptr, const std::vector<ir::TensorPtr> &output_tensor_vector,
                          int64_t chunks, int64_t dim) {
  const auto &cur_shape = input_tensor_ptr->Shape();
  const auto &cur_strides = GetTensorStrides(input_tensor_ptr);
  const auto cur_offset = input_tensor_ptr->StorageOffset();
  const auto ndim = cur_shape.size();
  CHECK_IF_FAIL_MSG(ndim > 0, "For 'Chunk', input's rank should be greater than 0, but got " + std::to_string(ndim));
  CHECK_IF_FAIL_MSG(chunks > 0, "For 'Chunk', chunks should be greater than 0, but got " + std::to_string(chunks));

  const auto wrap_dim = DynamicDimWrap(dim, ndim);
  const int64_t dim_size = cur_shape[wrap_dim];
  const int64_t split_size = (dim_size + chunks - 1) / chunks;
  if (MS_UNLIKELY(dim_size == 0)) {
    if (split_size == 0) {
      CHECK_IF_FAIL_MSG(static_cast<int64_t>(output_tensor_vector.size()) == chunks,
                        "For 'Chunk', output tensor size (" + std::to_string(output_tensor_vector.size()) +
                          ") does not match expected chunks (" + std::to_string(chunks) + ")");
      for (int64_t i = 0; i < chunks; ++i) {
        UpdateTensorViewInfo(input_tensor_ptr, output_tensor_vector[i], cur_shape, cur_strides);
      }
      return;
    }
    LOG_EXCEPTION << "For 'Chunk', output_num must be positive, but got 0";
  }

  // Calculate the number of sub tensors after segmentation
  const auto num_splits = std::max<int64_t>((dim_size + split_size - 1) / split_size, 1);
  const auto last_split_size = split_size - (split_size * num_splits - dim_size);
  CHECK_IF_FAIL_MSG(static_cast<int64_t>(output_tensor_vector.size()) == num_splits,
                    "For 'Chunk', output tensor size (" + std::to_string(output_tensor_vector.size()) +
                      ") does not match expected chunks (" + std::to_string(chunks) + ")");
  for (int64_t idx = 0; idx < num_splits; ++idx) {
    // Calculate the shape and length of sub tensors
    std::vector<int64_t> slice_shape = cur_shape;

    // Calculate the size of a sub tensor in a specified dimension
    slice_shape[wrap_dim] = (idx == num_splits - 1) ? last_split_size : split_size;
    // Calculate the storage offset of sub tensors
    const size_t new_storage_offset = cur_offset + LongToSize(idx * split_size * cur_strides[wrap_dim]);
    UpdateTensorViewInfo(input_tensor_ptr, output_tensor_vector[idx], slice_shape, cur_strides, new_storage_offset);
  }
}
}  // namespace

OpsErrorCode AclnnChunkView::CalcWorkspace(const std::vector<const ir::Value *> &input, const ir::Value *output,
                                           size_t *workspaceSize) {
  const auto input_tensor_ptr = input[kIndex0]->ToTensor();
  const auto chunks = input[kIndex1]->ToInt();
  CHECK_IF_FAIL_MSG(chunks >= 0, "chunks must be positive, but got " + std::to_string(chunks));
  const auto dim = input[kIndex2]->ToInt();
  UpdateOutputViewInfo(input_tensor_ptr, output->ToTuple()->ToTensorList(), chunks, dim);
  return SUCCESS;
}

MRT_REG_OP(chunk_view, AclnnChunkView, Ascend);
}  // namespace ops
}  // namespace mrt
