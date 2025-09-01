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

#include "common/common.h"
#include "ir/tensor/tensor.h"

namespace mrt {
namespace ir {

namespace {
int64_t CalculateNumel(const std::vector<int64_t> &shape, bool allow_dynamic) {
  int64_t numel = 1;
  for (const auto &dim : shape) {
    if (dim < 0) {
      if (allow_dynamic) {
        return -1;
      } else {
        throw std::runtime_error("Creating Tensor from existing data does not support dynamic shapes.");
      }
    }
    numel *= dim;
  }
  return numel;
}
}  // namespace

/**
 * @brief Computes the strides of the tensor based on its dimensions.
 * The strides are computed for a contiguous tensor in row-major order.
 * If the shape is dynamic, strides after the dynamic dimension will be -1.
 */
void Tensor::ComputeStrides() {
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

Tensor::Tensor(const std::vector<int64_t> &shape, DataType dtype, hardware::Device device)
    : dtype_(dtype), shape_(shape) {
  ComputeStrides();
  numel_ = CalculateNumel(shape_, true);
  size_t sizeBytes = 0;
  if (!HasDynamicShape()) {
    sizeBytes = numel_ * dtype_.GetSize();
  }

  storage_ = MakeIntrusive<Storage>(sizeBytes, device);
}

void Tensor::ResizeStorage() {
  CHECK_IF_NULL(storage_);
  size_t sizeBytes = 0;
  if (!HasDynamicShape()) {
    sizeBytes = numel_ * dtype_.GetSize();
  }

  storage_ = MakeIntrusive<Storage>(sizeBytes, storage_->GetDevice());
}

Tensor::Tensor(StoragePtr storage, DataType dtype, const std::vector<int64_t> &shape)
    : dtype_(dtype), shape_(shape), storage_(storage) {
  ComputeStrides();
  numel_ = CalculateNumel(shape_, true);
  if (!HasDynamicShape()) {
    if (storage_->SizeBytes() < numel_ * dtype_.GetSize()) {
      throw std::runtime_error("Storage size is smaller than required by tensor dimensions and data type.");
    }
  }
}

Tensor::Tensor(void *data, const std::vector<int64_t> &shape, DataType dtype, hardware::Device device)
    : dtype_(dtype), shape_(shape) {
  ComputeStrides();
  numel_ = CalculateNumel(shape_, false);
  size_t sizeBytes = numel_ * dtype_.GetSize();

  storage_ = MakeIntrusive<Storage>(data, sizeBytes, device);
}

void Tensor::SetShape(const std::vector<int64_t> &shape) {
  shape_ = shape;
  ComputeStrides();
  numel_ = CalculateNumel(shape_, true);
}

void Tensor::SetShape(const std::vector<int64_t> &&shape) {
  shape_ = std::move(shape);
  ComputeStrides();
  numel_ = CalculateNumel(shape_, true);
}

std::ostream &operator<<(std::ostream &os, Tensor *tensor) {
  if (tensor == nullptr) {
    os << "Null";
  } else {
    os << *tensor;
  }
  return os;
}

std::ostream &operator<<(std::ostream &os, const Tensor *tensor) {
  if (tensor == nullptr) {
    os << "Null";
  } else {
    os << *tensor;
  }
  return os;
}

std::ostream &operator<<(std::ostream &os, const Tensor &tensor) {
  constexpr size_t numelLimit = 30;
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
      if (numel <= numelLimit) {
        for (size_t i = 0; i < numel; ++i) {
          os << data[i];
          if (i < numel - 1) {
            os << ", ";
          }
        }
      } else {
        for (size_t i = 0; i < numelLimit; ++i) {
          os << data[i] << ", ";
        }
        os << "...";
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