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

#ifndef __OPS_ASCEND_ACLNN_UTILS_ACLNN_COMMON_META_H__
#define __OPS_ASCEND_ACLNN_UTILS_ACLNN_COMMON_META_H__

#include "acl/acl_base.h"

namespace mrt {
namespace ops {

// Base acl data structure
using aclOpExecutor = struct aclOpExecutor;
using aclTensor = struct aclTensor;
using aclTensorList = struct aclTensorList;
using aclScalar = struct aclScalar;
using aclIntArray = struct aclIntArray;
using aclFloatArray = struct aclFloatArray;
using aclBoolArray = struct aclBoolArray;

// Base acl creators
using _aclCreateTensorFuncPtr = aclTensor *(*)(const int64_t *viewDims, uint64_t viewDimsNum, aclDataType dataType,
                                               const int64_t *stride, int64_t offset, aclFormat format,
                                               const int64_t *storageDims, uint64_t storageDimsNum, void *tensorData);
using _aclCreateScalarFuncPtr = aclScalar *(*)(void *value, aclDataType dataType);
using _aclCreateIntArrayFuncPtr = aclIntArray *(*)(const int64_t *value, uint64_t size);
using _aclCreateFloatArrayFuncPtr = aclFloatArray *(*)(const float *value, uint64_t size);
using _aclCreateBoolArrayFuncPtr = aclBoolArray *(*)(const bool *value, uint64_t size);
using _aclCreateTensorListFuncPtr = aclTensorList *(*)(const aclTensor *const *value, uint64_t size);

// Base acl deleters
using _aclDestroyTensorFuncPtr = int (*)(const aclTensor *tensor);
using _aclDestroyScalarFuncPtr = int (*)(const aclScalar *scalar);
using _aclDestroyIntArrayFuncPtr = int (*)(const aclIntArray *array);
using _aclDestroyFloatArrayFuncPtr = int (*)(const aclFloatArray *array);
using _aclDestroyBoolArrayFuncPtr = int (*)(const aclBoolArray *array);
using _aclDestroyTensorListFuncPtr = int (*)(const aclTensorList *array);
using _aclDestroyAclOpExecutorFuncPtr = int (*)(aclOpExecutor *executor);

// Init and finalize
using _aclnnInitFuncPtr = int (*)(const char *);
using _aclnnFinalizeFuncPtr = int (*)();

// For reusing aclOpExecutor
using _aclSetAclOpExecutorRepeatableFuncPtr = int (*)(aclOpExecutor *executor);

// Set the device address ptr for aclTensor
using _aclSetTensorAddrFuncPtr = int (*)(aclOpExecutor *executor, const size_t index, aclTensor *tensor, void *addr);
using _aclSetDynamicTensorAddrFuncPtr = int (*)(aclOpExecutor *executor, const size_t index, const size_t relativeIndex,
                                                aclTensorList *tensors, void *addr);

#define DECLARE_ACLNN_COMMON_META_FUNC(name) _##name##FuncPtr name##_ = nullptr

#define EXTERN_ACLNN_COMMON_META_FUNC(name) \
  extern _##name##FuncPtr name##_;          \
  inline constexpr const char *kName##name##_ = #name

EXTERN_ACLNN_COMMON_META_FUNC(aclCreateTensor);
EXTERN_ACLNN_COMMON_META_FUNC(aclCreateScalar);
EXTERN_ACLNN_COMMON_META_FUNC(aclCreateIntArray);
EXTERN_ACLNN_COMMON_META_FUNC(aclCreateFloatArray);
EXTERN_ACLNN_COMMON_META_FUNC(aclCreateBoolArray);
EXTERN_ACLNN_COMMON_META_FUNC(aclCreateTensorList);

EXTERN_ACLNN_COMMON_META_FUNC(aclDestroyTensor);
EXTERN_ACLNN_COMMON_META_FUNC(aclDestroyScalar);
EXTERN_ACLNN_COMMON_META_FUNC(aclDestroyIntArray);
EXTERN_ACLNN_COMMON_META_FUNC(aclDestroyFloatArray);
EXTERN_ACLNN_COMMON_META_FUNC(aclDestroyBoolArray);
EXTERN_ACLNN_COMMON_META_FUNC(aclDestroyTensorList);
EXTERN_ACLNN_COMMON_META_FUNC(aclDestroyAclOpExecutor);

EXTERN_ACLNN_COMMON_META_FUNC(aclnnInit);
EXTERN_ACLNN_COMMON_META_FUNC(aclnnFinalize);

EXTERN_ACLNN_COMMON_META_FUNC(aclSetAclOpExecutorRepeatable);

EXTERN_ACLNN_COMMON_META_FUNC(aclSetTensorAddr);
EXTERN_ACLNN_COMMON_META_FUNC(aclSetDynamicTensorAddr);

#define GET_ACLNN_COMMON_META_FUNC(name) \
  []() -> auto {                         \
    if (name##_ == nullptr) {            \
      LoadOpApiLib();                    \
    }                                    \
    return name##_;                      \
  }                                      \
  ()

#define GET_ACLNN_OP_FUNC(name) GetAclnnOpApiFunc(name.c_str())

}  // namespace ops
}  // namespace mrt
#endif  // __OPS_ASCEND_ACLNN_UTILS_ACLNN_COMMON_META_H__
