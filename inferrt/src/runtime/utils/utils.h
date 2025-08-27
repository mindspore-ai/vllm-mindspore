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

#include "common/common.h"
#include "ir/graph.h"

namespace mrt {
namespace runtime {
constexpr size_t kFirstInput = 0;
constexpr size_t kSecondInput = 1;
extern const std::unordered_map<ops::Op, size_t> opsOutputFromInputIndex;
extern const std::unordered_map<ops::Op, size_t> opsOutputValueFromInputIndex;
extern const std::set<ops::Op> dummyOpsSet;
extern const std::set<ops::Op> forceResizeOpsSet;

inline bool IsSkipRecordRefCount(ir::NodePtr tensor) {
  CHECK_IF_NULL(tensor);
  return (tensor->op == ops::Op_End || tensor->op == ops::Op_load || tensor->op == ops::Op_update_state);
}

inline bool IsNodeOutputFromInput(ir::NodePtr tensor) {
  CHECK_IF_NULL(tensor);
  return opsOutputFromInputIndex.find(tensor->op) != opsOutputFromInputIndex.end();
}

inline bool IsDummyNode(ir::NodePtr node) {
  CHECK_IF_NULL(node);
  return dummyOpsSet.find(node->op) != dummyOpsSet.end();
}

inline bool IsSkipBuildDAKernel(ir::NodePtr node) {
  CHECK_IF_NULL(node);
  return (IsNodeOutputFromInput(node) || node->op == ops::Op_End || node->op == ops::Op_make_tuple ||
          node->op == ops::Op_tuple_getitem);
}

inline bool IsDAKernelNeedForceResize(ir::NodePtr node) {
  CHECK_IF_NULL(node);
  return forceResizeOpsSet.find(node->op) != forceResizeOpsSet.end();
}

}  // namespace runtime
}  // namespace mrt
#endif  // __RUNTIME_UTILS_H__
