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

#ifndef __OPS_ASCEND_ATB_ATB_LINEAR_H__
#define __OPS_ASCEND_ATB_ATB_LINEAR_H__

#include "ops/ascend/atb/atb_base.h"

namespace mrt {
namespace ops {

class MRT_EXPORT AtbLinear : public AtbBase {
 public:
  AtbLinear() : AtbBase("linear") {}
  ~AtbLinear() override = default;

  OpsErrorCode CalcWorkspace(const std::vector<const ir::Value *> &inputs, const ir::Value *output,
                             size_t *workspace_size) override;
  OpsErrorCode Launch(const std::vector<const ir::Value *> &inputs, void *workspace, size_t workspaceSize,
                      ir::Value *output, void *stream) override;

 private:
  bool IsBiasNone(const std::vector<const ir::Value *> &inputs) const {
    return inputs.size() <= 2 || inputs[2] == nullptr || inputs[2]->IsNone();
  }
};

}  // namespace ops
}  // namespace mrt

#endif  // __OPS_ASCEND_ATB_ATB_LINEAR_H__
