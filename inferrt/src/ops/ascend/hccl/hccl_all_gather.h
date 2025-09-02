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

#ifndef OPS_ASCEND_HCCL_ALL_GATHER_H_
#define OPS_ASCEND_HCCL_ALL_GATHER_H_

#include <vector>

#include "ops/op_base/op_all_gather.h"

#include "ops/operator.h"
#include "ops/ascend/hccl/hccl_kernel.h"

namespace mrt {
namespace ops {
class HcclAllGather : public OpAllGather {
 public:
  HcclAllGather() = default;
  ~HcclAllGather() = default;

  OpsErrorCode CalcWorkspace(const std::vector<const ir::Value *> &input, const ir::Value *output,
                             size_t *workspace_size) override;
  OpsErrorCode Launch(const std::vector<const ir::Value *> &input, void *workspace, size_t workspaceSize,
                      ir::Value *output, void *stream) override;

 private:
  HcclKernel hcclKernel;
};
}  // namespace ops
}  // namespace mrt
#endif  // OPS_ASCEND_HCCL_ALL_GATHER_H_
