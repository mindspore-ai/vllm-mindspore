/**
 * Copyright 2025 liu xu
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

#ifndef __TENSOR_TENSOR_DATA_H__
#define __TENSOR_TENSOR_DATA_H__

#include <memory>
#include <string>

#include "common/common.h"

#ifndef DA_TENSOR_MAX_DIM
#define DA_TENSOR_MAX_DIM 6
#endif

namespace da {
namespace tensor {
// Data type of tensor
enum Type {
  Type_Tuple,
  Type_Monad,
  Type_Tensor,
  Type_None,
  Type_Bool,
  Type_F16,
  Type_F32,
  Type_F64,
  Type_I16,
  Type_I32,
  Type_I64,
  Type_BF16,
  Type_End
};

enum TensorType { HOST_TENSOR, DEVICE_TENSOR, UNKNOW_TENSOR };

// Shape value type of tensor
using ShapeValueType = size_t;
using ShapeArray = ShapeValueType[DA_TENSOR_MAX_DIM];

// Get element size according to tensor shape
inline ssize_t ShapeSize(const ShapeArray &shape) {
  ssize_t size = 1;
  for (auto dim : shape) {
    if (dim < 0) {
      return 0;  // dynamic shape scenario
    }

    if (dim == 0) {
      break;
    }
    size *= static_cast<ssize_t>(dim);
  }
  return size;
}

// Get dims according to shape array
inline size_t ShapeDims(const ShapeArray &shape) {
  size_t dims = 0;
  for (auto dim : shape) {
    if (dim == 0) {
      return dims;
    }
    ++dims;
  }
  return dims;
}

inline size_t DataTypeSize(da::tensor::Type dtype) {
  switch (dtype) {
    case da::tensor::Type_Bool:
      return sizeof(bool);
    case da::tensor::Type_F32:
      return sizeof(float);
    case da::tensor::Type_F64:
      return sizeof(double);
    case da::tensor::Type_I16:
      return sizeof(int16_t);
    case da::tensor::Type_I32:
      return sizeof(int32_t);
    case da::tensor::Type_I64:
      return sizeof(int64_t);
    default:
      LOG_ERROR << "Unknown data type: " << dtype;
      exit(EXIT_FAILURE);
  }
}

struct TensorData {
  Type dtype{Type_End};
};

template <typename T>
struct TensorDataImpl : public TensorData {
  size_t ndim{0};
  ssize_t nbytes{0};
  ssize_t size{0};
  bool fromMemPool{false};
  T *data{nullptr};
};
}  // namespace tensor
}  // namespace da
#endif  // __TENSOR_TENSOR_DATA_H__
