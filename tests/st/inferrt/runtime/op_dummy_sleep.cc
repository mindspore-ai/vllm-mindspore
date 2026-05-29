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

#include <chrono>
#include <thread>
#include "include/custom_op_api.h"

namespace mrt {
namespace ops {
class DummySleepOp : public AclnnCustomOperator {
 public:
  DummySleepOp() : AclnnCustomOperator("aclnnAdd") {}
  ~DummySleepOp() override = default;

  OpsErrorCode CalcWorkspace(const std::vector<const ir::Value *> &input, const ir::Value *output,
                             size_t *workspaceSize) override {
    GetExecutor()->GetWorkspaceSize(reinterpret_cast<uint64_t *>(workspaceSize), input[0]->ToTensor(),
                                    input[1]->ToTensor(), input[2], output->ToTensor());
    return SUCCESS;
  }

  OpsErrorCode Launch(const std::vector<const ir::Value *> &input, void *workspace, size_t workspaceSize,
                      ir::Value *output, void *stream) override {
    GetExecutor()->Launch(workspace, workspaceSize, stream, input[0]->ToTensor(), input[1]->ToTensor(), input[2],
                          output->ToTensor());
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    return SUCCESS;
  }
};

REGISTER_CUSTOM_OP(dummy_sleep, DummySleepOp);
}  // namespace ops
}  // namespace mrt
