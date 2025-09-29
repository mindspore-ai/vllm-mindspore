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

#ifndef __OPS_ASCEND_ACLNN_UTILS_ACLNN_CONVERTER_H__
#define __OPS_ASCEND_ACLNN_UTILS_ACLNN_CONVERTER_H__

#include <cstddef>
#include <utility>
#include <vector>
#include <tuple>
#include <string>
#include <type_traits>

#include "ir/value/value.h"
#include "ops/utils/op_constants.h"
#include "ops/ascend/aclnn/utils/convert_utils.h"
#include "ops/ascend/aclnn/utils/aclnn_common_meta.h"
#include "ops/ascend/aclnn/utils/opapi_lib_loader.h"

namespace mrt {
namespace ops {
// Convert dtype
aclDataType Convert(ir::DataType::Type dtype);

// Convert tensor
inline aclTensor *Convert(const ir::TensorPtr &tensor) {
  static const auto aclCreateTensor = GET_ACLNN_COMMON_META_FUNC(aclCreateTensor);
  CHECK_IF_NULL(aclCreateTensor);
  if (tensor == nullptr || tensor->Dtype().value == ir::DataType::Type::Unknown) {
    return nullptr;
  }
  aclFormat format = ACL_FORMAT_ND;
  switch (tensor->Dim()) {
    case kDim3:
      format = ACL_FORMAT_NCL;
      break;
    case kDim4:
      format = ACL_FORMAT_NCHW;
      break;
    case kDim5:
      format = ACL_FORMAT_NCDHW;
      break;
    default:
      format = ACL_FORMAT_ND;
  }
  if (tensor->Format() == ir::MemoryFormat::FRACTAL_NZ) {
    format = ACL_FORMAT_FRACTAL_NZ;
  }
  return aclCreateTensor(tensor->Shape().data(), tensor->Dim(), Convert(tensor->Dtype().value),
                         tensor->Strides().data(), tensor->StorageOffset(), format, tensor->Shape().data(),
                         tensor->Dim(), tensor->DataPtr());
}

inline aclTensorList *Convert(const std::vector<ir::TensorPtr> &tensorList) {
  if (tensorList.empty()) {
    LOG_OUT << "tensorList is empty";
  }
  static const auto aclCreateTensorList = GET_ACLNN_COMMON_META_FUNC(aclCreateTensorList);
  std::vector<aclTensor *> aclTensorList;
  for (const auto &tensor : tensorList) {
    (void)aclTensorList.emplace_back(Convert(tensor));
  }
  return aclCreateTensorList(aclTensorList.data(), aclTensorList.size());
}

// Convert scalar
template <typename T, typename = std::enable_if_t<std::is_scalar_v<T>>>
T Convert(T value) {
  return value;
}

inline const char *Convert(const std::string &str) { return str.c_str(); }

inline aclIntArray *Convert(const std::vector<int64_t> &intList) {
  static const auto aclCreateIntArray = GET_ACLNN_COMMON_META_FUNC(aclCreateIntArray);
  CHECK_IF_NULL(aclCreateIntArray);
  return aclCreateIntArray(intList.data(), intList.size());
}

inline aclBoolArray *Convert(const std::vector<uint8_t> &boolList) {
  static const auto aclCreateBoolArray = GET_ACLNN_COMMON_META_FUNC(aclCreateBoolArray);
  CHECK_IF_NULL(aclCreateBoolArray);
  return aclCreateBoolArray(reinterpret_cast<const bool *>(boolList.data()), boolList.size());
}

inline aclFloatArray *Convert(const std::vector<float> &floatList) {
  static const auto aclCreateFloatArray = GET_ACLNN_COMMON_META_FUNC(aclCreateFloatArray);
  CHECK_IF_NULL(aclCreateFloatArray);
  return aclCreateFloatArray(floatList.data(), floatList.size());
}

inline void *Convert(const ir::TuplePtr &tuple) {
  if (tuple->Size() == 0) {
    LOG_OUT << "tuple is empty";
    return nullptr;
  }
  auto firstElement = (*tuple)[kIndex0];
  // Not support tuple in tuple and the element type in the tuple must be the same.
  if (firstElement->IsTensor()) {
    std::vector<ir::TensorPtr> tensorList;
    TupleToTensorList(*tuple, &tensorList);
    return Convert(tensorList);
  } else if (firstElement->IsInt()) {
    std::vector<int64_t> intList;
    TupleToIntList(*tuple, &intList);
    return Convert(intList);
  } else if (firstElement->IsFloat()) {
    std::vector<float> floatList;
    TupleToFloatList(*tuple, &floatList);
    return Convert(floatList);
  } else if (firstElement->IsBool()) {
    std::vector<uint8_t> boolList;
    TupleToBoolList(*tuple, &boolList);
    return Convert(boolList);
  } else {
    LOG_EXCEPTION << "Invalid element type in tuple: " << tuple;
  }
}

// Main entry for convert
template <typename... Args>
constexpr auto ConvertParams(const Args &...args) {
  return std::make_tuple(Convert(args)...);
}

}  // namespace ops
}  // namespace mrt
#endif  // __OPS_ASCEND_ACLNN_UTILS_ACLNN_CONVERTER_H__
