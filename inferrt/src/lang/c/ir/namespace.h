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

#ifndef __PARSER_IR_NAMESPACE_H__
#define __PARSER_IR_NAMESPACE_H__

#include <iomanip>
#include <string>
#include <vector>

#include "lang/c/ir/node.h"

namespace da {
namespace ir {
class Namespace {
 public:
  Namespace() = default;
  virtual ~Namespace() = default;

  void setName(const std::string &name) { name_ = name; }
  const std::string &name() const { return name_; }

  const std::vector<ComplexNode> &nodes() const { return nodes_; };
  void AddNode(const ComplexNode &node) { nodes_.emplace_back(node); };
  void AddNode(const ComplexNode &&node) { nodes_.emplace_back(std::move(node)); };

  virtual const std::string ToString() const { return ""; }

 protected:
  std::string name_;
  std::vector<ComplexNode> nodes_;
  // Namespaces namespaces_;
};
using NsPtr = Namespace *;
using NsConstPtr = const Namespace *;
using Namespaces = std::vector<NamespacePtr>;

class Block : public Namespace {
 public:
  Block() = default;
  virtual ~Block() = default;

 protected:
  std::vector<Node> freeVariables_;
};
using BlockPtr = Block *;
using Blocks = std::vector<BlockPtr>;

class Func : public Namespace {
 public:
  Func() = default;
  virtual ~Func() = default;

  void AddParameter(const std::string &paramName) { parameters_.emplace_back(Param(paramName)); }
  void AddParameter(const std::string &paramName, const std::string &defaultParam) {
    parameters_.emplace_back(Param(paramName, defaultParam));
  }
  const Params &parameters() { return parameters_; }

  // void AddNs(NsConstPtr ns) { namespaces_.emplace_back(ns); }
  // const Namespaces &namespaces() { return namespaces_; }

  const std::string ToString() const override {
    std::stringstream ss;
    ss << name_;
    ss << '(';
    for (const auto &p : parameters_) {
      ss << p.ToString() << ',';
    }
    ss.seekp(-1, std::ios_base::end);
    ss << ')';
    return ss.str();
  }

 protected:
  Params parameters_;
};
using FuncPtr = Func *;
using Funcs = std::vector<FuncPtr>;

NamespacePtr NewNamespace();
void ClearNamespacePool();

BlockPtr NewBlock();
void ClearBlockPool();

FuncPtr NewFunc();
void ClearFuncPool();
}  // namespace ir
}  // namespace da

#endif  // __PARSER_IR_NAMESPACE_H__
