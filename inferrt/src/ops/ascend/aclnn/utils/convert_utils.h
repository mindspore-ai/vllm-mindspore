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

#ifndef __OPS_ASCEND_ACLNN_UTILS_CONVERT_UTILS_H__
#define __OPS_ASCEND_ACLNN_UTILS_CONVERT_UTILS_H__

#include <vector>
#include "ir/value/value.h"

namespace mrt {
namespace ops {
inline void TupleToTensorList(const ir::Tuple &tuple, std::vector<ir::TensorPtr> *tensorList) {
  for (size_t i = 0; i < tuple.Size(); ++i) {
    (void)tensorList->emplace_back(tuple[i]->ToTensor());
  }
}

inline void TupleToIntList(const ir::Tuple &tuple, std::vector<int64_t> *intList) {
  for (size_t i = 0; i < tuple.Size(); ++i) {
    (void)intList->emplace_back(tuple[i]->ToInt());
  }
}

inline void TupleToBoolList(const ir::Tuple &tuple, std::vector<uint8_t> *boolList) {
  for (size_t i = 0; i < tuple.Size(); ++i) {
    (void)boolList->emplace_back(static_cast<uint8_t>(tuple[i]->ToBool()));
  }
}

inline void TupleToFloatList(const ir::Tuple &tuple, std::vector<float> *floatList) {
  for (size_t i = 0; i < tuple.Size(); ++i) {
    (void)floatList->emplace_back(tuple[i]->ToFloat());
  }
}

inline void TupleToDoubleList(const ir::Tuple &tuple, std::vector<double> *doubleList) {
  for (size_t i = 0; i < tuple.Size(); ++i) {
    (void)doubleList->emplace_back(tuple[i]->ToDouble());
  }
}

}  // namespace ops
}  // namespace mrt
#endif  // __OPS_ASCEND_ACLNN_UTILS_CONVERT_UTILS_H__
