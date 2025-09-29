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

#ifndef __OPS_ASCEND_ACLNN_UTILS_ACLNN_DELETER_H__
#define __OPS_ASCEND_ACLNN_UTILS_ACLNN_DELETER_H__

#include <utility>
#include <tuple>

#include "ops/ascend/aclnn/utils/opapi_lib_loader.h"

namespace mrt {
namespace ops {
// Release tensor
inline void Release(aclTensor *tensor) {
  static const auto aclDestroyTensor = GET_ACLNN_COMMON_META_FUNC(aclDestroyTensor);
  CHECK_IF_NULL(aclDestroyTensor);
  aclDestroyTensor(tensor);
}

inline void Release(aclTensorList *tensorList) {
  static const auto aclDestroyTensorList = GET_ACLNN_COMMON_META_FUNC(aclDestroyTensorList);
  CHECK_IF_NULL(aclDestroyTensorList);
  aclDestroyTensorList(tensorList);
}

// Release scalar
inline void Release(aclIntArray *intList) {
  static const auto aclDestroyIntArray = GET_ACLNN_COMMON_META_FUNC(aclDestroyIntArray);
  CHECK_IF_NULL(aclDestroyIntArray);
  aclDestroyIntArray(intList);
}

inline void Release(aclBoolArray *boolList) {
  static const auto aclDestroyBoolArray = GET_ACLNN_COMMON_META_FUNC(aclDestroyBoolArray);
  CHECK_IF_NULL(aclDestroyBoolArray);
  aclDestroyBoolArray(boolList);
}

inline void Release(aclFloatArray *floatList) {
  static const auto aclDestroyFloatArray = GET_ACLNN_COMMON_META_FUNC(aclDestroyFloatArray);
  CHECK_IF_NULL(aclDestroyFloatArray);
  aclDestroyFloatArray(floatList);
}

template <typename T>
void Release(T value) {
  (void)value;
}

// Main entry for release converted params
template <typename Tuple>
void ReleaseConvertedParams(const Tuple &t) {
  static constexpr auto size = std::tuple_size<Tuple>::value;
  CallRelease(t, std::make_index_sequence<size>{});
}

template <typename Tuple, size_t... I>
void CallRelease(const Tuple &t, std::index_sequence<I...>) {
  (Release(std::get<I>(t)), ...);
}

// Release executor
inline void ReleaseExecutor(aclOpExecutor *executor) {
  static const auto aclDestroyAclOpExecutor = GET_ACLNN_COMMON_META_FUNC(aclDestroyAclOpExecutor);
  if (aclDestroyAclOpExecutor == nullptr) {
    LOG_OUT << "aclDestroyAclOpExecutor is nullptr";
    return;
  }
  auto ret = aclDestroyAclOpExecutor(executor);
  if (ret != 0) {
    LOG_EXCEPTION << "aclDestroyAclOpExecutor failed";
  }
}

}  // namespace ops
}  // namespace mrt
#endif  // __OPS_ASCEND_ACLNN_UTILS_ACLNN_DELETER_H__
