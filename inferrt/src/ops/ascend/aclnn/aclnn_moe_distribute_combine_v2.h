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

#ifndef __OPS_ASCEND_ACLNN_ACLNN_MOE_DISTRIBUTE_COMBINE_V2_H__
#define __OPS_ASCEND_ACLNN_ACLNN_MOE_DISTRIBUTE_COMBINE_V2_H__

#include "ops/operator.h"
#include "ops/ascend/aclnn/utils/aclnn_executor.h"

namespace mrt {
namespace ops {
class AclnnMoeDistributeCombineV2 : public Operator {
 public:
  AclnnMoeDistributeCombineV2();
  ~AclnnMoeDistributeCombineV2() override = default;

  OpsErrorCode CalcWorkspace(const std::vector<const ir::Value *> &input, const ir::Value *output,
                             size_t *workspaceSize) override;
  OpsErrorCode Launch(const std::vector<const ir::Value *> &input, void *workspace, size_t workspaceSize,
                      ir::Value *output, void *stream) override;

 private:
  bool use_v4_{false};
  bool use_v3_{false};
  std::unique_ptr<AclnnExecutor> executor_v4_{nullptr};
  std::unique_ptr<AclnnExecutor> executor_v2_{nullptr};
  std::unique_ptr<AclnnExecutor> executor_v3_{nullptr};
  AclnnExecutor *active_executor_{nullptr};
};

}  // namespace ops
}  // namespace mrt
#endif  // __OPS_ASCEND_ACLNN_ACLNN_MOE_DISTRIBUTE_COMBINE_V2_H__
