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

#include "pass/pass.h"

namespace da {
namespace pass {
inline DATensor *NodePass::NewTensor(ops::Op op, DATensor **start,
                                     size_t size) {
  return PassManager::Instance().NewTensor(op, start, size);
}

inline DATensor *NodePass::NewTensor(ops::Op op,
                                     const std::vector<DATensor *> &inputs) {
  return PassManager::Instance().NewTensor(op, inputs);
}

void PassManager::Run(DAGraph *graph, TensorCreator &creator) {
  if (passes_.empty() || graph->nodeSize == 0) {
    LOG_OUT << "No pass or no node in graph. instance: " << this;
    return;
  }
  LOG_OUT << "Start running passes.";
  tensorCreator_ = creator;
  // Initialize the tensor list and map.
  orderedNodes_ = OrderedNodes(graph);
  ud_ = UserDef(graph);

  // Do transform for each node.
  auto tensors = orderedNodes_.tensorList();
  for (const DATensor *node : tensors) {
    LOG_OUT << "Handle node: " << ToString(node);
    for (auto &pass : passes_) {
      LOG_OUT << "Handle pass '" << pass.first << "'";
      if (!pass.second->Match(node)) {
        continue;
      }
      LOG_OUT << "Matched pass '" << pass.first << "'";
      auto newNode = pass.second->Replacement();
      CHECK_IF_NULL(newNode);
      CHECK_IF_FAIL(Replace(node, newNode));
    }
  }

  // Flush optimized ordered nodes into graph.
  orderedNodes_.Flush(graph);
}

bool PassManager::Replace(const DATensor *oldNode, const DATensor *newNode) {
  LOG_OUT << "To replace " << ToString(oldNode) << " to " << ToString(newNode)
          << ", nodes size: " << orderedNodes_.tensorList().size();
  auto users = ud_.FindUsers(oldNode);
  if (users.empty()) {
    LOG_ERROR << "No user for node: " << ToString(oldNode);
    return false;
  }
  LOG_OUT << "users size: " << users.size();

  // Replace old node with new node.
  // We add node firstly, then remove old node.
  for (auto &user : users) {
    DATensor *owner = const_cast<DATensor *>(user.first);
    InsertOrderedNodes(owner, user.second, oldNode, newNode);
    RemoveOrderedNodes(owner, user.second, oldNode);
    owner->input[user.second] = const_cast<DATensor *>(newNode);
  }
  LOG_OUT << "Finish replace, nodes size: "
          << orderedNodes_.tensorList().size();

  // Since we run add&delete, now we can free unused tensor safely.
  for (auto &unused : unusedList_) {
    // TODO: free all unused tensors.
  }
  return true;
}

void PassManager::RemoveOrderedNodes(const DATensor *owner, size_t index,
                                     const DATensor *node) {
  if (!ud_.DropNode(owner, index, node)) { // 'node' has other users.
    LOG_OUT << "Has other users, " << ToString(node);
    return;
  }
  // 'node' has no user anymore.
  LOG_OUT << "Run real remove, " << ToString(node);
  (void)unusedList_.emplace_back(node);
  if (!orderedNodes_.Remove(node)) {
    return;
  }
  for (size_t i = 0; i < node->inputSize; ++i) {
    RemoveOrderedNodes(node, i, node->input[i]);
  }
}

void PassManager::InsertOrderedNodes(const DATensor *owner, size_t index,
                                     const DATensor *anchor,
                                     const DATensor *node) {
  if (!ud_.AddNode(owner, index, node)) { // 'node' is not first insertion.
    LOG_OUT << "Has other users, " << ToString(node);
    return;
  }
  if (!orderedNodes_.Insert(anchor, node)) {
    LOG_OUT << "Insert failed, " << ToString(node);
    return;
  }
  LOG_OUT << "Insert for inputs, " << ToString(node);
  for (size_t i = 0; i < node->inputSize; ++i) {
    InsertOrderedNodes(node, i, anchor, node->input[i]);
  }
}

bool PassManager::OrderedNodes::Insert(const DATensor *anchor,
                                       const DATensor *node) {
  if (tensorMap_.count(node) != 0) {
    return false; // Already exists.
  }
  auto iter = tensorMap_.find(anchor);
  if (iter == tensorMap_.cend()) {
    return false; // Not found anchor.
  }
  auto listIter = tensorList_.emplace(iter->second, node);
  (void)tensorMap_.emplace(node, listIter);
  return true;
}

bool PassManager::OrderedNodes::Append(const DATensor *node) {
  if (tensorMap_.find(node) != tensorMap_.cend()) {
    return false;
  }
  auto iter = tensorList_.emplace(tensorList_.end(), node);
  (void)tensorMap_.emplace(node, iter);
  return true;
}

bool PassManager::OrderedNodes::Remove(const DATensor *node) {
  auto iter = tensorMap_.find(node);
  if (iter == tensorMap_.cend()) {
    return false;
  }
  tensorList_.erase(iter->second);
  tensorMap_.erase(iter);
  return true;
}

void PassManager::OrderedNodes::Init(DAGraph *graph) {
  for (ssize_t i = graph->nodeSize - 1; i >= 0; --i) {
    auto tensor = graph->node[i];
    auto iter = tensorList_.emplace(tensorList_.end(), tensor);
    (void)tensorMap_.emplace(tensor, iter);
  }
}

void PassManager::OrderedNodes::Flush(DAGraph *graph) {
  graph->nodeSize = 0;
  for (auto iter = tensorList_.crbegin(); iter != tensorList_.crend(); ++iter) {
    graph->node[graph->nodeSize] = const_cast<DATensor *>(*iter);
    ++graph->nodeSize;
  }
}

class ManualSamplePass : public NodePass {
public:
  // If node is matched.
  bool Match(const DATensor *node) override {
    node_ = node;
    LOG_OUT << "To match " << ToString(node);
    return node->op == ops::Op_add;
  };

  // Replacement node for the matched node.
  DATensor *Replacement() override {
    return NewTensor(ops::Op_mul, const_cast<DATensor **>(node_->input),
                     node_->inputSize);
  };

private:
  const DATensor *node_;
};

DA_REGISTER_PASS("ManualSamplePass", ManualSamplePass);
} // namespace pass
} // namespace da
