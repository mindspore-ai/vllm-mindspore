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

#ifndef __OPS_ASCEND_ACLNN_ACLNN_TRANSPOSE_BATCH_MATMUL_H__
#define __OPS_ASCEND_ACLNN_ACLNN_TRANSPOSE_BATCH_MATMUL_H__

#include <cstdint>
#include <optional>
#include <vector>

#include "ops/ascend/aclnn/utils/aclnn_executor.h"
#include "ops/operator.h"

namespace mrt {
namespace ops {
class AclnnTransposeBatchMatmul : public Operator {
 public:
  AclnnTransposeBatchMatmul() { executor_ = std::make_unique<AclnnExecutor>("aclnnTransposeBatchMatMul"); }
  ~AclnnTransposeBatchMatmul() override = default;

  OpsErrorCode CalcWorkspace(const std::vector<const ir::Value *> &input, const ir::Value *output,
                             size_t *workspaceSize) override;
  OpsErrorCode Launch(const std::vector<const ir::Value *> &input, void *workspace, size_t workspaceSize,
                      ir::Value *output, void *stream) override;

 private:
  void ParseAndValidateInputs(const std::vector<const ir::Value *> &input);

  std::unique_ptr<AclnnExecutor> executor_{nullptr};
  ir::TensorPtr input_{nullptr};
  ir::TensorPtr weight_{nullptr};
  std::optional<ir::TensorPtr> bias_;
  std::optional<ir::TensorPtr> scale_;
  std::vector<int64_t> permX1_;
  std::vector<int64_t> permX2_;
  std::vector<int64_t> permY_;
  int8_t cubeMathType_{0};
  int32_t batchSplitFactor_{1};
};

}  // namespace ops
}  // namespace mrt

#endif  // __OPS_ASCEND_ACLNN_ACLNN_TRANSPOSE_BATCH_MATMUL_H__
