#ifndef __PARSER_IR_NODE_H__
#define __PARSER_IR_NODE_H__

#include <vector>

#include "ir/operator.h"

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
} // namespace ir

#endif // __PARSER_IR_NODE_H__