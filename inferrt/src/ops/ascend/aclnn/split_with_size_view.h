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

#ifndef __OPS_ASCEND_ACLNN_ACLNN_SPLIT_WITH_SIZE_VIEW_H__
#define __OPS_ASCEND_ACLNN_ACLNN_SPLIT_WITH_SIZE_VIEW_H__

#include "ops/ascend/aclnn/view_base.h"
#include "ops/ascend/aclnn/utils/aclnn_executor.h"

namespace mrt {
namespace ops {
class AclnnSplitWithSizeView : public AclnnViewBase {
 public:
  AclnnSplitWithSizeView() = default;
  ~AclnnSplitWithSizeView() override = default;

  OpsErrorCode CalcWorkspace(const std::vector<const ir::Value *> &input, const ir::Value *output,
                             size_t *workspaceSize) override;
};

}  // namespace ops
}  // namespace mrt
#endif  // __OPS_ASCEND_ACLNN_ACLNN_SPLIT_WITH_SIZE_VIEW_H__
