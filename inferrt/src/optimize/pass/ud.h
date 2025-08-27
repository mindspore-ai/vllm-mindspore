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

#ifndef __OPTIMIZE_PASS_UD_H__
#define __OPTIMIZE_PASS_UD_H__

#include <utility>
#include <list>
#include <unordered_map>

#include "common/common.h"
#include "ir/graph.h"

namespace mrt {
namespace pass {
class UserDef {
 public:
  using UserList = std::list<std::pair<ir::NodePtr, size_t>>;

  UserDef() = default;
  explicit UserDef(ir::GraphPtr graph) : graph_{graph} { BuildUD(); }

  void BuildUD() {
    root_ = graph_->nodes.back();
    for (ssize_t i = graph_->nodes.size() - 1; i >= 0; --i) {
      const auto node = graph_->nodes[i];
      CHECK_IF_NULL(node);
      for (size_t j = 0; j < node->inputs.size(); ++j) {
        const auto input = node->inputs[j];
        // CHECK_IF_NULL(input);
        auto iter = users_.find(input);
        if (iter == users_.cend()) {
          LOG_OUT << "Find " << input << " first user " << node << " at index " << j;
          (void)users_.emplace(input, UserList({std::make_pair(node, j)}));
        } else {
          LOG_OUT << "Find " << input << " user " << node << " at index " << j;
          (void)iter->second.emplace_back(std::make_pair(node, j));
        }
      }
    }
  }

  // Add 'node' as a input of 'owner'.
  // Return true if first user, or false otherwise.
  bool AddNode(const ir::NodePtr owner, size_t index, const ir::NodePtr node) {
    // Add user 'node' for 'owner'
    auto iter = users_.find(node);
    if (iter == users_.cend()) {
      (void)users_.emplace(node, std::list({std::make_pair(owner, index)}));
      return true;
    } else {
      (void)iter->second.emplace_back(std::make_pair(owner, index));
      return false;
    }
  }

  void AddNodes(const ir::NodePtr owner, size_t index, const ir::NodePtr node) {
    if (!AddNode(owner, index, node)) {
      return;  // Maybe an old node, no need to traverse its inputs.
    }
    // Add users for 'node'.
    for (size_t i = 0; i < node->inputs.size(); ++i) {
      const auto input = node->inputs[i];
      AddNodes(node, i, input);
    }
  }

  // Drop 'node' from 'owner'.
  // Return true if erased(no user anymore), or false otherwise.
  bool DropNode(const ir::NodePtr owner, size_t index, const ir::NodePtr node) {
    auto iter = users_.find(node);
    if (iter == users_.cend()) {
      LOG_ERROR << "'node' has no user.";
      return true;
    }
    (void)iter->second.remove(std::make_pair(owner, index));
    if (!iter->second.empty()) {
      return false;
    }
    users_.erase(iter);
    return true;
  }

  void DropNodes(const ir::NodePtr owner, size_t index, const ir::NodePtr node) {
    if (!DropNode(owner, index, node)) {
      return;
    }
    // Continue to drop inputs.
    for (size_t i = 0; i < node->inputs.size(); ++i) {
      const auto input = node->inputs[i];
      DropNodes(node, i, input);
    }
  }

  // Drop a node.
  void DropNode(const ir::NodePtr node) {
    for (size_t i = 0; i < node->inputs.size(); ++i) {
      const auto input = node->inputs[i];
      auto iter = users_.find(input);
      if (iter == users_.cend()) {
        continue;
      } else {
        (void)iter->second.remove(std::make_pair(node, i));
        if (iter->second.empty()) {
          users_.erase(iter);
          // Continue to drop its inputs below.
        } else {
          continue;
        }
      }
      // Continue to drop inputs.
      DropNode(input);
    }
  }

  UserList FindUsers(const ir::NodePtr node) {
    if (users_.count(node) == 0) {
      return UserList();
    }
    return users_[node];
  }

  ir::NodePtr Root() const { return root_; }
  bool IsRoot(const ir::NodePtr node) const { return root_ == node; }

 private:
  ir::GraphPtr graph_;
  ir::NodePtr root_;
  std::unordered_map<ir::NodePtr, UserList> users_;
};
}  // namespace pass
}  // namespace mrt

#endif  // __OPTIMIZE_PASS_UD_H__