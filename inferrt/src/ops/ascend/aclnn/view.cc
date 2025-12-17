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

#include "ops/ascend/aclnn/view.h"

#include <vector>
#include "ops/ascend/aclnn/utils/opapi_utils.h"
#include "ops/ascend/aclnn/utils/view_utils.h"
#include "ops/op_register.h"

namespace mrt {
namespace ops {
namespace {
std::optional<std::vector<int64_t>> CalculateViewStrides(const std::vector<int64_t> &cur_shape,
                                                         const std::vector<int64_t> &cur_strides,
                                                         const std::vector<int64_t> &new_shape) {
  if (cur_shape.empty()) {
    return std::vector<int64_t>(new_shape.size(), 1);
  }

  bool is_old_empty = std::any_of(cur_shape.begin(), cur_shape.end(), [](const int64_t dim) { return dim == 0; });
  if (is_old_empty && cur_shape == new_shape) {
    return cur_strides;
  }

  const int64_t new_rank = SizeToLong(new_shape.size());
  std::vector<int64_t> new_strides(new_rank, 0);
  if (is_old_empty) {
    for (int64_t dim = new_rank - 1; dim >= 0; --dim) {
      if (dim == (new_rank - 1)) {
        new_strides[dim] = 1;
      } else {
        new_strides[dim] = std::max(new_shape[dim + 1], static_cast<int64_t>(1)) * new_strides[dim + 1];
      }
    }
    return new_strides;
  }

  int64_t view_dim = new_rank - 1;
  int64_t base_stride = cur_strides.back();
  int64_t tensor_elems = 1;
  int64_t view_elems = 1;
  for (int64_t dim = SizeToLong(cur_shape.size()) - 1; dim >= 0; --dim) {
    tensor_elems *= cur_shape[dim];
    if (dim == 0 || (cur_shape[dim - 1] != 1 && cur_strides[dim - 1] != tensor_elems * base_stride)) {
      while (view_dim >= 0 && (view_elems < tensor_elems || new_shape[view_dim] == 1)) {
        new_strides[view_dim] = view_elems * base_stride;
        view_elems *= new_shape[view_dim];
        --view_dim;
      }
      if (view_elems != tensor_elems) {
        return std::nullopt;
      }
      if (dim > 0) {
        base_stride = cur_strides[dim - 1];
        tensor_elems = 1;
        view_elems = 1;
      }
    }
  }
  if (view_dim != -1) {
    return std::nullopt;
  }

  return new_strides;
}

std::vector<int64_t> InferSizeImpl(const std::vector<int64_t> &new_shape, int64_t num_elements) {
  int64_t new_size = 1;
  std::optional<int64_t> infer_dim;
  for (int64_t dim = 0, ndim = static_cast<int64_t>(new_shape.size()); dim != ndim; ++dim) {
    if (new_shape[dim] == -1) {
      if (infer_dim) {
        LOG_EXCEPTION << "only one dimension can be inferred";
      }
      infer_dim = dim;
    } else if (new_shape[dim] >= 0) {
      new_size *= new_shape[dim];
    } else {
      LOG_EXCEPTION << "invalid proposed_shape dimension";
    }
  }

  if (num_elements == new_size || (infer_dim && new_size > 0 && num_elements % new_size == 0)) {
    std::vector<int64_t> res(new_shape);
    if (infer_dim) {
      if (new_size == 0) {
        LOG_OUT << "WARNING: cannot reshape tensor of 0 elements into proposed_shape, because the unspecified "
                   "dimension size -1 can be any value and is ambiguous";
        res[*infer_dim] = 0;
      } else {
        res[*infer_dim] = num_elements / new_size;
      }
    }
    return res;
  }
  LOG_EXCEPTION << "proposed_shape is invalid for input of size";
  return {};
}

std::vector<int64_t> InferShape(const std::vector<int64_t> &new_shape, const std::vector<int64_t> &cur_shape) {
  const int64_t num_elements =
    std::accumulate(cur_shape.begin(), cur_shape.end(), static_cast<int64_t>(1), std::multiplies<int64_t>());
  auto res = InferSizeImpl(new_shape, num_elements);
  return res;
}

void UpdateOutputViewInfo(const ir::TensorPtr &input_tensor_ptr, const ir::TensorPtr &output_tensor_ptr,
                          const std::vector<int64_t> &new_shape) {
  const auto &cur_shape = input_tensor_ptr->Shape();
  const auto &cur_strides = GetTensorStrides(input_tensor_ptr);
  const auto infer_shape = InferShape(new_shape, cur_shape);
  const auto strides = CalculateViewStrides(cur_shape, cur_strides, infer_shape);
  if (strides.has_value()) {
    UpdateTensorViewInfo(input_tensor_ptr, output_tensor_ptr, infer_shape, strides.value());
    return;
  }
  LOG_EXCEPTION << "View shape " << new_shape << "is not compatible with input tensor's shape " << cur_shape
                << " and stride " << cur_strides;
}
}  // namespace

OpsErrorCode AclnnView::CalcWorkspace(const std::vector<const ir::Value *> &input, const ir::Value *output,
                                      size_t *workspaceSize) {
  const auto input_tensor_ptr = input[kIndex0]->ToTensor();
  if (!input_tensor_ptr->IsContiguous()) {
    LOG_EXCEPTION << "Input tensor is not contiguous";
  }
  const auto &shape = input[kIndex1]->ToTuple()->ToIntList();
  if (std::any_of(shape.begin(), shape.end(), [](const int &shape_i) { return shape_i < -1; })) {
    LOG_EXCEPTION << "For View the component of shape can't be less than -1";
  }
  UpdateOutputViewInfo(input_tensor_ptr, output->ToTensor(), shape);
  return SUCCESS;
}

MRT_REG_OP(view, AclnnView, Ascend);
}  // namespace ops
}  // namespace mrt
