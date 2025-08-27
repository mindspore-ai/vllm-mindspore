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

#include "runtime/utils/utils.h"
#include <set>
#include <unordered_map>
#include <vector>

namespace mrt {
namespace runtime {
const std::unordered_map<ops::Op, size_t> opsOutputFromInputIndex = {
  {ops::Op_return, kFirstInput},
  {ops::Op_depend, kFirstInput},
  {ops::Op_load, kFirstInput},
  {ops::Op_update_state, kFirstInput},
};

const std::unordered_map<ops::Op, size_t> opsOutputValueFromInputIndex = {
  {ops::Op_reshape_ext, kFirstInput},
};

const std::set<ops::Op> dummyOpsSet = {
  ops::Op_tuple_getitem,
  ops::Op_depend,
  ops::Op_make_tuple,
  ops::Op_reshape_ext,
};

const std::set<ops::Op> forceResizeOpsSet = {
  ops::Op_flash_attention_score,
  ops::Op_paged_attention,
};

void GetNodeRealInputs(ir::NodePtr node) {
  CHECK_IF_NULL(node);
  std::vector<ir::NodePtr > realInputs;
  for (auto input : node->inputs) {
    CHECK_IF_NULL(input);
    if (input->output.IsTuple()) {
      auto elements = input->output.ToTuple().GetElements();
      for (const auto &element : elements) {
        auto fakeNode = std::make_shared<ir::Node>(); // TODO: 
        fakeNode->output = element;
        (void)realInputs.emplace_back(fakeNode);
      }
    } else {
      (void)realInputs.emplace_back(input);
    }
  }
  node->inputs = realInputs;
}
}  // namespace runtime
}  // namespace mrt
