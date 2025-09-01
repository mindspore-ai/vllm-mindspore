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

#ifndef __OPS_ASCEND_ACLNN_UTILS_ACLNN_HASH_H__
#define __OPS_ASCEND_ACLNN_UTILS_ACLNN_HASH_H__

#include <string>
#include <vector>

#include "ir/value/value.h"
#include "ops/ascend/aclnn/utils/hash_buf.h"

namespace mrt {
namespace ops {
// Gather tensor
void GatherHash(const ir::TensorPtr &tensor);

inline void GatherHash(const std::vector<ir::TensorPtr> &tensorList) {
  for (auto &tensor : tensorList) {
    GatherHash(tensor);
  }
}

// Gather scalar, int64_t/bool/float/double, etc.
template <typename T>
void GatherHash(const T &value) {
  MemcpyToBuf(&value, sizeof(T));
}

// Gather vector scalar.
template <typename T>
void GatherHash(const std::vector<T> &values) {
  MemcpyToBuf(values.data(), values.size() * sizeof(T));
}

inline void GatherHash(const std::string &str) { MemcpyToBuf(str.c_str(), str.size()); }

// Gather value
inline void GatherHash(const ir::ValuePtr &value) {
  if (value == nullptr || value->IsNone()) {
    return;
  }
  if (value->IsTensor()) {
    GatherHash(value->ToTensor());
  } else if (value->IsInt()) {
    GatherHash(value->ToInt());
  } else if (value->IsFloat()) {
    GatherHash(value->ToFloat());
  } else if (value->IsBool()) {
    GatherHash(value->ToBool());
  } else if (value->IsString()) {
    GatherHash(value->ToString());
  } else {
    LOG_EXCEPTION << "Invalid value type: " << value << " for hash from tuple";
  }
}

// Gather tuple
inline void GatherHash(const ir::TuplePtr &tuple) {
  if (tuple == nullptr || tuple->Size() == 0) {
    return;
  }
  for (const auto &value : *tuple) {
    GatherHash(value);
  }
}

inline void GatherHash() {}

template <typename T, typename... Args>
void GatherHash(const T &arg, const Args &...args) {
  GatherHash(arg);
  GatherHash(args...);
}

// Main entry for calculate hash
template <typename... Args>
uint64_t CalcAclnnHash(const std::string &opName, const Args &...args) {
  gHashOffset = 0;
  GatherHash(opName, args...);
  return CalcHashId();
}

}  // namespace ops
}  // namespace mrt
#endif  // __OPS_ASCEND_ACLNN_UTILS_ACLNN_HASH_H__
