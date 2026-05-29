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

#include "include/custom_op_api.h"

namespace mrt {
namespace ops {
class DummyThrowOp : public Operator {
 public:
  DummyThrowOp() = default;
  ~DummyThrowOp() override = default;

  OpsErrorCode InferShape(const std::vector<const ir::Value *> &input, ir::Value *output) override {
    output->ToTensor()->SetShape(input[0]->ToTensor()->Shape());
    return SUCCESS;
  }

  OpsErrorCode CalcWorkspace(const std::vector<const ir::Value *> &input, const ir::Value *output,
                             size_t *workspaceSize) override {
    *workspaceSize = 0;
    return SUCCESS;
  }

  OpsErrorCode Launch(const std::vector<const ir::Value *> &input, void *workspace, size_t workspaceSize,
                      ir::Value *output, void *stream) override {
    if (++launchCount_ > 1) {
      LOG_EXCEPTION << "DummyThrowOp intentionally throws in second Launch for testing";
    }
    return SUCCESS;
  }

 private:
  size_t launchCount_ = 0;
};

REGISTER_CUSTOM_OP(dummy_throw, DummyThrowOp);
}  // namespace ops
}  // namespace mrt
