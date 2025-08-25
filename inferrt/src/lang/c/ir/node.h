/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#ifndef __PARSER_IR_NODE_H__
#define __PARSER_IR_NODE_H__

#include <vector>

#include "lang/c/ir/operator.h"

namespace da {
namespace ir {
class Namespace;
using Ns = Namespace;
using NamespacePtr = Namespace *;
using NsPtr = Namespace *;
using NsConstPtr = const Namespace *;

class Node {
 public:
  Node() = default;
  Node(const std::string &name) : name_{name} {}

  virtual const std::string &name() const { return name_; }
  virtual const std::string ToString() const { return name_; }

 protected:
  std::string name_;
};
using NodePtr = Node *;
using NodeConstPtr = const Node *;
using Nodes = std::vector<NodeConstPtr>;

class ParameterNode : public Node {
 public:
  ParameterNode() = default;
  ParameterNode(const std::string &name, const std::string &defaultParam = "")
      : Node(name), defaultParam_{defaultParam} {}

  const std::string ToString() const override {
    if (!defaultParam_.empty()) {
      return name_ + '=' + defaultParam_;
    }
    return name_;
  }

 private:
  std::string defaultParam_;
};
using Param = ParameterNode;
using ParamPtr = Param *;
using Params = std::vector<Param>;

class ValueNode : public Node {};
using VNode = ValueNode;
using VNodePtr = VNode *;

class OperatorNode : public ValueNode {
 private:
  Op op_;
};
using OpNode = OperatorNode;
using OpNodePtr = OpNode *;

class NamespaceNode : public ValueNode {
 private:
  NsPtr ns_;
};
using NsNode = NamespaceNode;
using NsNodePtr = NsNode *;

class ComplexNode : public Node {
  ComplexNode() = delete;
  ComplexNode(std::initializer_list<NodeConstPtr> &nodes) : inputs_{nodes} {}

  const Nodes &inputs() const { return inputs_; };

 private:
  Nodes inputs_;
};
using CNode = ComplexNode;
using CNodePtr = CNode *;
}  // namespace ir
}  // namespace da

#endif  // __PARSER_IR_NODE_H__