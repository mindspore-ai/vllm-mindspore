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

#ifndef __OPS_ASCEND_MEM_MEMCPY_LIKES_H__
#define __OPS_ASCEND_MEM_MEMCPY_LIKES_H__

#include "ops/operator.h"
#include "ops/ascend/aclnn/utils/aclnn_executor.h"

namespace mrt {
namespace ops {
class MemcpyOpBase : public Operator {
 public:
  MemcpyOpBase() = default;
  ~MemcpyOpBase() override = default;

  OpsErrorCode CalcWorkspace(const std::vector<const ir::Value *> &input, const ir::Value *output,
                             size_t *workspaceSize) override;
  OpsErrorCode Launch(const std::vector<const ir::Value *> &input, void *workspace, size_t workspaceSize,
                      ir::Value *output, void *stream) override;
  std::vector<std::pair<uint32_t, uint32_t>> GetOutputInputRefPairs() const override {
    return {std::pair<uint32_t, uint32_t>(0, 0)};
  }
  bool NeedLaunch() override;
};

#define DefineMemcpyOp(op_name)         \
  class op_name : public MemcpyOpBase { \
   public:                              \
    op_name() {}                        \
    ~op_name() override = default;      \
  }

// view memcpy ops
DefineMemcpyOp(Flatten);
DefineMemcpyOp(Unsqueeze);
}  // namespace ops
}  // namespace mrt
#endif  // __OPS_ASCEND_MEM_MEMCPY_LIKES_H__
