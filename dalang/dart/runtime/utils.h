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

#ifndef __RUNTIME_UTILS_H__
#define __RUNTIME_UTILS_H__

#include <set>
#include <unordered_map>

#include "tensor/tensor.h"

namespace da {
namespace runtime {
using namespace tensor;
namespace {
constexpr size_t kFirstInput = 0;
constexpr size_t kSecondInput = 1;

const std::unordered_map<ops::Op, size_t> opsOutputFromInputIndex = {
    {ops::Op_return, kFirstInput},
    {ops::Op_depend, kFirstInput},
    {ops::Op_load, kFirstInput},
    {ops::Op_update_state, kFirstInput},
};

const std::set<ops::Op> dummyDATensorNodeSet = {
    ops::Op_tuple_getitem,
    ops::Op_depend,
    ops::Op_make_tuple,
    ops::Op_return,
};
} // namespace

template <typename T>
const T GetValue(const DATensor *tensor) {
  if (tensor->tensorType != HOST_TENSOR || tensor->dim != 0) {
    LOG_ERROR << "Input DATensor is not HOST_TENSOR or is not a scalar";
    exit(EXIT_FAILURE);
  }

  const T *dataPtr = reinterpret_cast<const T *>(tensor->data);
  LOG_OUT << "GetValue for DATensor: " << tensor << ", dataPtr: " << dataPtr;
  CHECK_IF_NULL(dataPtr);
  return *dataPtr;
}

void GetNodeRealInputs(DATensor *node);

inline void CloneDATensorShape(DATensor *dstNode, DATensor *sourceNode) {
  CHECK_IF_NULL(dstNode);
  CHECK_IF_NULL(sourceNode);
  dstNode->dim = sourceNode->dim;
  for (size_t i = 0; i < dstNode->dim; ++i) {
    dstNode->shape[i] = sourceNode->shape[i];
  }
}

inline bool IsSkipRecordRefCount(DATensor *tensor) {
  CHECK_IF_NULL(tensor);
  return (tensor->op == ops::Op_End || tensor->op == ops::Op_load ||
          tensor->op == ops::Op_update_state);
}

inline bool IsDATensorOutputFromInput(DATensor *tensor) {
  CHECK_IF_NULL(tensor);
  return opsOutputFromInputIndex.find(tensor->op) !=
         opsOutputFromInputIndex.end();
}

inline size_t GetDATensorOuputFromInputIndex(DATensor *node) {
  CHECK_IF_NULL(node);
  auto iter = opsOutputFromInputIndex.find(node->op);
  if (iter == opsOutputFromInputIndex.end()) {
    LOG_ERROR << "Can not find ops." << ops::ToStr(node->op)
              << " in opsOutputFromInputIndex";
  }
  return iter->second;
}

inline bool IsDummyDATensorNode(DATensor *node) {
  CHECK_IF_NULL(node);
  return dummyDATensorNodeSet.find(node->op) != dummyDATensorNodeSet.end();
}
} // namespace runtime
} // namespace da
#endif // __RUNTIME_UTILS_H__
