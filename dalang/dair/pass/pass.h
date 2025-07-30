/**
 * Copyright 2025 Zhang Qinghua
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

#ifndef __PASS_PASS_H__
#define __PASS_PASS_H__

#include <functional>
#include <iterator>
#include <list>
#include <string.h>
#include <vector>

#include "tensor/tensor.h"
#include "tensor/ud.h"

namespace da {
namespace pass {
using namespace tensor;

using TensorCreator = std::function<DATensor *(ops::Op, DATensor **, size_t)>;

class NodePass {
public:
  // If node is matched.
  virtual bool Match(const DATensor *node) = 0;

  // Replacement node for the matched node.
  virtual DATensor *Replacement() = 0;

  DATensor *NewTensor(ops::Op op, DATensor **start, size_t size);
  DATensor *NewTensor(ops::Op op, const std::vector<DATensor *> &inputs);
};

class PassManager {
public:
  PassManager() = default;

  static PassManager &Instance() {
    static PassManager instance;
    LOG_OUT << "PassManager instance: '" << &instance << "'";
    return instance;
  }

  void Run(DAGraph *graph, TensorCreator &creator);

  void AddPass(const std::string &name, const NodePass &pass) {
    LOG_OUT << "Add pass '" << name << "'";
    (void)passes_.emplace_back(
        std::make_pair(name, const_cast<NodePass *>(&pass)));
  }

  DATensor *NewTensor(ops::Op op, DATensor **start, size_t size) {
    CHECK_IF_FAIL(tensorCreator_);
    return tensorCreator_(op, start, size);
  }

  DATensor *NewTensor(ops::Op op, const std::vector<DATensor *> &inputs) {
    CHECK_IF_FAIL(tensorCreator_);
    return tensorCreator_(op, const_cast<DATensor **>(inputs.data()),
                          inputs.size());
  }

private:
  using TensorList = std::list<const DATensor *>;

  bool Replace(const DATensor *oldNode, const DATensor *newNode);
  void RemoveOrderedNodes(const DATensor *owner, size_t index,
                          const DATensor *node);
  void InsertOrderedNodes(const DATensor *owner, size_t index,
                          const DATensor *anchor, const DATensor *node);

  class OrderedNodes {
  public:
    OrderedNodes() = default;
    OrderedNodes(DAGraph *graph) { Init(graph); }

    // Return true if inserted, or false otherwise.
    bool Insert(const DATensor *anchor, const DATensor *node);
    // Return true if appended, or false otherwise.
    bool Append(const DATensor *node);
    // Return true if erased, or false otherwise.
    bool Remove(const DATensor *node);
    // Build the ordered nodes.
    void Init(DAGraph *graph);
    // Flush the nodes back into graph.
    void Flush(DAGraph *graph);

    const TensorList &tensorList() const { return tensorList_; }

  private:
    TensorList tensorList_;
    std::unordered_map<const DATensor *, TensorList::iterator> tensorMap_;
  } orderedNodes_;

  std::vector<std::pair<std::string, NodePass *>> passes_;
  UserDef ud_;
  std::vector<const DATensor *> unusedList_;
  TensorCreator tensorCreator_;
};

class PassRegister {
public:
  PassRegister(const std::string &name, NodePass &pass) {
    LOG_OUT << "Register pass '" << name;
    PassManager::Instance().AddPass(name, pass);
  }
};

#define DA_REGISTER_PASS(PASS_NAME, PASS)                                      \
  static PASS __pass_##PASS##__;                                               \
  static const PassRegister __passRegister_##PASS##__(PASS_NAME,               \
                                                      __pass_##PASS##__);
} // namespace pass
} // namespace da

#endif // __PASS_PASS_H__