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

#ifndef __OPS_CPU_ATEN_UTILS_ATEN_CONVERTER_H__
#define __OPS_CPU_ATEN_UTILS_ATEN_CONVERTER_H__

#include <torch/torch.h>

#include "common/logger.h"
#include "ir/common/dtype.h"
#include "ir/value/value.h"

namespace mrt {
namespace ops {
inline at::ScalarType ToAtenDType(ir::DataType type) {
  switch (type) {
    case ir::DataType::Bool:
      return at::kBool;
    case ir::DataType::Float32:
      return at::kFloat;
    case ir::DataType::Float64:
      return at::kDouble;
    case ir::DataType::Int16:
      return at::kShort;
    case ir::DataType::Int32:
      return at::kInt;
    case ir::DataType::Int64:
      return at::kLong;
    default:
      LOG_ERROR << "Unsupported DataType for Aten conversion.";
      exit(EXIT_FAILURE);
  }
}

inline at::Tensor ToAtenTensor(const ir::Value *value) {
  auto tensor = value->ToTensor();
  auto options = at::TensorOptions().dtype(ToAtenDType(tensor->Dtype()));
  return at::from_blob(const_cast<void *>(tensor->DataPtr()), tensor->Shape(), options);
}

}  // namespace ops
}  // namespace mrt
#endif  // __OPS_CPU_ATEN_UTILS_ATEN_CONVERTER_H__
