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

#ifndef __OPS_ASCEND_ACLNN_UTILS_VIEW_UTILS_H__
#define __OPS_ASCEND_ACLNN_UTILS_VIEW_UTILS_H__

#include <cstdint>
#include <vector>
#include "common/visible.h"
#include "ir/tensor/tensor.h"
#include "ir/value/value.h"

namespace mrt {
namespace ops {
DA_API std::vector<int64_t> CalculateStrides(const std::vector<int64_t> &shape);

DA_API int64_t DynamicDimWrap(int64_t dim, int64_t dim_post_expr, bool wrap_scalar = false);

DA_API std::vector<int64_t> GetTensorStrides(const ir::TensorPtr &tensor_ptr);

DA_API void UpdateTensorViewInfo(const ir::TensorPtr &input_tensor_ptr, const ir::TensorPtr &output_tensor_ptr,
                                 const std::vector<int64_t> &new_shape, const std::vector<int64_t> &new_strides,
                                 size_t new_storage_offset);

DA_API inline void UpdateTensorViewInfo(const ir::TensorPtr &input_tensor_ptr, const ir::TensorPtr &output_tensor_ptr,
                                        const std::vector<int64_t> &new_shape,
                                        const std::vector<int64_t> &new_strides) {
  UpdateTensorViewInfo(input_tensor_ptr, output_tensor_ptr, new_shape, new_strides, input_tensor_ptr->StorageOffset());
}

DA_API std::vector<std::pair<uint32_t, uint32_t>> GenerateOutputInputRefPair(const ir::Value *output);
}  // namespace ops
}  // namespace mrt
#endif  //__OPS_ASCEND_ACLNN_UTILS_VIEW_UTILS_H__
