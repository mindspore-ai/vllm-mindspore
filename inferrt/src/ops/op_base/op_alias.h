/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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

#ifndef __OPS_OP_BASE_OP_ALIAS_H__
#define __OPS_OP_BASE_OP_ALIAS_H__

#include <vector>

#include "ops/operator.h"

namespace mrt {
namespace ops {
class OpAlias : public Operator {
 public:
  OpAlias() = default;
  ~OpAlias() override = default;

  OpsErrorCode Launch(const std::vector<const ir::Value *> &input, void *workspace, size_t workspaceSize,
                      ir::Value *output, void *stream) override {
    return SUCCESS;
  }

  std::vector<std::pair<uint32_t, uint32_t>> GetOutputInputRefPairs() const override {
    return {std::pair<uint32_t, uint32_t>(0, 0)};
  }

  bool NeedLaunch() override { return false; }
};
}  // namespace ops
}  // namespace mrt
#endif  // __OPS_OP_BASE_OP_ALIAS_H__
