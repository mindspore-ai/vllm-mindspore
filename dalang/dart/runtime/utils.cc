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

#include <vector>

#include "runtime/utils.h"

namespace da {
namespace runtime {
void GetNodeRealInputs(DATensor *node) {
  CHECK_IF_NULL(node);
  std::vector<DATensor *> realInputs;
  for (size_t i = 0; i < node->inputSize; ++i) {
    CHECK_IF_NULL(node->input[i]);
    if (node->input[i]->type == Type_Tensor) {
      auto **tensorList = static_cast<DATensor **>(node->input[i]->data);
      CHECK_IF_NULL(tensorList);
      for (size_t j = 0; j < node->input[i]->shape[0]; j++) {
        CHECK_IF_NULL(tensorList[j]);
        if (tensorList[j]->type == Type_Monad) {
          continue;
        }
        (void)realInputs.emplace_back(tensorList[j]);
      }
      continue;
    }
    if (node->input[i]->type == Type_Monad) {
      continue;
    }
    (void)realInputs.emplace_back(node->input[i]);
  }
  node->inputSize = realInputs.size();
  for (size_t i = 0; i < node->inputSize; ++i) {
    node->input[i] = realInputs[i];
  }
}
} // namespace runtime
} // namespace da
