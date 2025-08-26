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

#include <stdexcept>
#include <numeric>
#include <sstream>

#include "ir/tensor/tensor.h"

namespace mrt {
namespace ir {

/**
 * @brief Constructs a TensorImpl.
 * @param storage The underlying storage for the tensor data.
 * @param dtype The data type of the tensor elements.
 * @param shape The dimensions of the tensor.
 * @throws std::runtime_error if the storage size is insufficient for the given dimensions and data type.
 */
TensorImpl::TensorImpl(Storage storage, DataType dtype, const std::vector<int64_t> &shape)
    : storage_(std::move(storage)), dtype_(dtype), shape_(shape) {
  bool isDynamic = false;
  numel_ = 1;
  for (const auto &dim : shape) {
    if (dim < 0) {
      isDynamic = true;
      break;
    }
    numel_ *= dim;
  }

  if (isDynamic) {
    numel_ = -1;
  }

  if (!isDynamic) {
    if (storage_.SizeBytes() < numel_ * dtype_.GetSize()) {
      throw std::runtime_error("Storage size is smaller than required by tensor dimensions and data type.");
    }
  }
  ComputeStrides();
}

/**
 * @brief Computes the strides of the tensor based on its dimensions.
 * The strides are computed for a contiguous tensor in row-major order.
 * If the shape is dynamic, strides after the dynamic dimension will be -1.
 */
void TensorImpl::ComputeStrides() {
  if (shape_.empty()) {
    return;
  }
  strides_.resize(shape_.size());
  int64_t stride = 1;
  for (int i = shape_.size() - 1; i >= 0; --i) {
    strides_[i] = stride;
    if (stride != -1) {
      if (shape_[i] < 0) {
        stride = -1;
      } else {
        stride *= shape_[i];
      }
    }
  }
}

/**
 * @brief Creates an empty tensor with uninitialized data.
 * A new storage is allocated for the tensor.
 * @param shape The dimensions of the tensor.
 * @param dtype The data type of the tensor.
 * @param device The device to allocate the tensor on.
 * @return The newly created tensor.
 */
Tensor Empty(const std::vector<int64_t> &shape, DataType dtype, hardware::Device device) {
  int64_t numel = 1;
  bool isDynamic = false;
  for (const auto &dim : shape) {
    if (dim < 0) {
      isDynamic = true;
      break;
    }
    numel *= dim;
  }

  size_t sizeBytes = 0;
  if (!isDynamic) {
    sizeBytes = numel * dtype.GetSize();
  }

  Storage storage(sizeBytes, device);
  return Tensor(storage, dtype, shape);
}

/**
 * @brief Creates a tensor from an existing data blob.
 * The tensor does not own the memory.
 * @param data Pointer to the data.
 * @param shape The dimensions of the tensor.
 * @param dtype The data type of the tensor.
 * @param device The device where the data is located.
 * @return The newly created tensor.
 */
Tensor FromBlob(void *data, const std::vector<int64_t> &shape, DataType dtype, hardware::Device device) {
  int64_t numel = 1;
  for (const auto &dim : shape) {
    if (dim < 0) {
      throw std::runtime_error("FromBlob does not support dynamic shapes.");
    }
    numel *= dim;
  }

  size_t sizeBytes = numel * dtype.GetSize();

  Storage storage(data, sizeBytes, device);
  return Tensor(storage, dtype, shape);
}

std::ostream &operator<<(std::ostream &os, const Tensor &tensor) {
  if (!tensor.Defined()) {
    os << "Tensor(undefined)";
    return os;
  }
  os << "Tensor(shape=[";
  const auto &shape = tensor.Shape();
  for (size_t i = 0; i < shape.size(); ++i) {
    os << shape[i];
    if (i < shape.size() - 1) {
      os << ", ";
    }
  }
  os << "], dtype=" << tensor.Dtype().ToString();
  os << ", data=[";
  if (tensor.DataPtr()) {
    if (tensor.Dtype() == DataType::Float32) {  // TODO: support other dtypes
      const auto data = static_cast<const float *>(tensor.DataPtr());
      const size_t numel = tensor.Numel();
      for (size_t i = 0; i < numel; ++i) {
        os << data[i];
        if (i < numel - 1) {
          os << ", ";
        }
      }
    } else {
      os << "...";
    }
  } else {
    os << "null";
  }
  os << "])";
  return os;
}

}  // namespace ir
}  // namespace mrt