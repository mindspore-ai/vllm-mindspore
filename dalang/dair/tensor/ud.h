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

#ifndef __TENSOR_UD_H__
#define __TENSOR_UD_H__

#include <list>
#include <unordered_map>

#include "tensor/tensor.h"

namespace da {
namespace tensor {
class UserDef {
public:
  using UserList = std::list<std::pair<const DATensor *, size_t>>;

  UserDef() = default;
  UserDef(const DAGraph *graph) : graph_{graph} { BuildUD(); }

  void BuildUD() {
    root_ = graph_->node[graph_->nodeSize - 1];
    for (ssize_t i = graph_->nodeSize - 1; i >= 0; --i) {
      const auto tensor = graph_->node[i];
      CHECK_IF_NULL(tensor);
      for (size_t j = 0; j < tensor->inputSize; ++j) {
        const auto input = tensor->input[j];
        // CHECK_IF_NULL(input);
        auto iter = users_.find(input);
        if (iter == users_.cend()) {
          LOG_OUT << "Find " << ToString(input) << " first user "
                  << ToString(tensor) << " at index " << j;
          (void)users_.emplace(input, UserList({std::make_pair(tensor, j)}));
        } else {
          LOG_OUT << "Find " << ToString(input) << " user " << ToString(tensor)
                  << " at index " << j;
          (void)iter->second.emplace_back(std::make_pair(tensor, j));
        }
      }
    }
  }

  // Add 'tensor' as a input of 'owner'.
  // Return true if first user, or false otherwise.
  bool AddNode(const DATensor *owner, size_t index, const DATensor *tensor) {
    // Add user 'tensor' for 'owner'
    auto iter = users_.find(tensor);
    if (iter == users_.cend()) {
      (void)users_.emplace(tensor, std::list({std::make_pair(owner, index)}));
      return true;
    } else {
      (void)iter->second.emplace_back(std::make_pair(owner, index));
      return false;
    }
  }

  void AddNodes(const DATensor *owner, size_t index, const DATensor *tensor) {
    if (!AddNode(owner, index, tensor)) {
      return; // Maybe an old tensor, no need to traverse its inputs.
    }
    // Add users for 'tensor'.
    for (size_t i = 0; i < tensor->inputSize; ++i) {
      const auto input = tensor->input[i];
      AddNodes(tensor, i, input);
    }
  }

  // Drop 'tensor' from 'owner'.
  // Return true if erased(no user anymore), or false otherwise.
  bool DropNode(const DATensor *owner, size_t index, const DATensor *tensor) {
    auto iter = users_.find(tensor);
    if (iter == users_.cend()) {
      LOG_ERROR << "'tensor' has no user.";
      return true;
    }
    (void)iter->second.remove(std::make_pair(owner, index));
    if (!iter->second.empty()) {
      return false;
    }
    users_.erase(iter);
    return true;
  }

  void DropNodes(const DATensor *owner, size_t index, const DATensor *tensor) {
    if (!DropNode(owner, index, tensor)) {
      return;
    }
    // Continue to drop inputs.
    for (size_t i = 0; i < tensor->inputSize; ++i) {
      const auto input = tensor->input[i];
      DropNodes(tensor, i, input);
    }
  }

  // Drop a node.
  void DropNode(const DATensor *tensor) {
    for (size_t i = 0; i < tensor->inputSize; ++i) {
      const auto input = tensor->input[i];
      auto iter = users_.find(input);
      if (iter == users_.cend()) {
        continue;
      } else {
        (void)iter->second.remove(std::make_pair(tensor, i));
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

  UserList FindUsers(const DATensor *tensor) {
    if (users_.count(tensor) == 0) {
      return UserList();
    }
    return users_[tensor];
  }

  const DATensor *Root() const { return root_; }
  bool IsRoot(const DATensor *node) const { return root_ == node; }

private:
  const DAGraph *graph_;
  const DATensor *root_;
  std::unordered_map<const DATensor *, UserList> users_;
};
} // namespace tensor
} // namespace da

#endif // __TENSOR_UD_H__