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

#ifndef __IR_GRAPH_H__
#define __IR_GRAPH_H__

#include <string>
#include <vector>
#include <memory>
#include <sstream>

#include "ir/common/intrusive_ptr.h"
#include "ir/value/value.h"
#include "ops/op_def/ops_name.h"

namespace mrt {
namespace ir {

/**
 * @brief Represents a node in the computation graph.
 *
 * A node corresponds to an operation, with a set of inputs and a single output.
 */
struct Node : public RefCounted {
  ops::Op op;                              ///< The operation performed by this node.
  std::vector<IntrusivePtr<Node>> inputs;  ///< The input nodes to the operation.
  ValuePtr output{nullptr};                ///< The output value from the operation.
};

/**
 * @brief Represents the entire computation graph.
 */
struct Graph : public RefCounted {
  std::vector<IntrusivePtr<Node>> nodes;  ///< The list of all nodes in the graph.
};

using NodePtr = IntrusivePtr<Node>;
using GraphPtr = IntrusivePtr<Graph>;

inline std::ostream &operator<<(std::ostream &os, const Node &node) {
  os << "Node("
     << "op=" << ops::ToStr(node.op) << ", value=" << node.output << ")";
  return os;
}

inline std::ostream &operator<<(std::ostream &os, const NodePtr &node) {
  if (node == nullptr) {
    os << "Null";
  } else {
    os << *node;
  }
  return os;
}

}  // namespace ir
}  // namespace mrt

#endif  // __IR_GRAPH_H__
