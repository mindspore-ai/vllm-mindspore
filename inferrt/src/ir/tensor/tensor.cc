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
#include <iomanip>

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
        LOG_EXCEPTION << "Creating Tensor from existing data does not support dynamic shapes.";
      }
    }
    numel *= dim;
  }
  return numel;
}

template <typename T>
void PrintData(std::ostream &os, const void *data, size_t numel, size_t limit) {
  const auto *d = static_cast<const T *>(data);
  for (size_t i = 0; i < std::min(numel, limit); ++i) {
    // Promote char types to int for printing
    os << +d[i];
    if (i < std::min(numel, limit) - 1) {
      os << ", ";
    }
  }
  if (numel > limit) {
    os << ", ...";
  }
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
  CHECK_IF_FAIL(!HasDynamicShape());
  size_t sizeBytes = numel_ * dtype_.GetSize();
  storage_->Resize(sizeBytes);
}

void Tensor::UpdateData(void *data) {
  storage_->SetData(data);
}

Tensor::Tensor(StoragePtr storage, const std::vector<int64_t> &shape, DataType dtype)
    : dtype_(dtype), shape_(shape), storage_(storage) {
  ComputeStrides();
  numel_ = CalculateNumel(shape_, true);
  if (!HasDynamicShape()) {
    if (storage_->SizeBytes() < numel_ * dtype_.GetSize()) {
      LOG_EXCEPTION << "Storage size is smaller than required by tensor dimensions and data type.";
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

std::ostream &operator<<(std::ostream &os, const TensorPtr &tensor) {
  if (!tensor) {
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
    if (tensor.HasDynamicShape()) {
      os << "dynamic shape, not materialized";
    } else if (tensor.Numel() > 0) {
      switch (tensor.Dtype()) {
        case DataType::Float32:
          PrintData<float>(os, tensor.DataPtr(), tensor.Numel(), numelLimit);
          break;
        case DataType::Float64:
          PrintData<double>(os, tensor.DataPtr(), tensor.Numel(), numelLimit);
          break;
        case DataType::Int8:
          PrintData<int8_t>(os, tensor.DataPtr(), tensor.Numel(), numelLimit);
          break;
        case DataType::Int16:
          PrintData<int16_t>(os, tensor.DataPtr(), tensor.Numel(), numelLimit);
          break;
        case DataType::Int32:
          PrintData<int32_t>(os, tensor.DataPtr(), tensor.Numel(), numelLimit);
          break;
        case DataType::Int64:
          PrintData<int64_t>(os, tensor.DataPtr(), tensor.Numel(), numelLimit);
          break;
        case DataType::UInt8:
          PrintData<uint8_t>(os, tensor.DataPtr(), tensor.Numel(), numelLimit);
          break;
        case DataType::Bool:
          os << std::boolalpha;
          PrintData<bool>(os, tensor.DataPtr(), tensor.Numel(), numelLimit);
          break;
        default:
          os << "...";
          break;
      }
    }
  } else {
    os << "null";
  }
  os << "])";
  return os;
}
}  // namespace ir
}  // namespace mrt
