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

#ifndef __OPS_TORCH_OP_TORCH_CALL_H__
#define __OPS_TORCH_OP_TORCH_CALL_H__

#include <torch/torch.h>
#include <string>
#include <vector>
#include "ops/op_register.h"
#include "ir/value/value.h"

namespace mrt {
namespace ops {
class OpTorchCall : public Operator {
 public:
  explicit OpTorchCall(const std::string &opName) {
    qualifiedOpName_ = opName;
    auto pos = qualifiedOpName_.find(".");
    if (pos != std::string::npos) {
      qualifiedOpName_.replace(pos, 1, "::");
    }
  }
  ~OpTorchCall() override = default;

  void Init(const std::vector<const ir::Value *> &inputs, const ir::Value *output);

  OpsErrorCode CalcWorkspace(const std::vector<const ir::Value *> &input, const ir::Value *output,
                             size_t *workspaceSize);
  OpsErrorCode Launch(const std::vector<const ir::Value *> &input, void *workspace, size_t workspaceSize,
                      ir::Value *output, void *stream);

 protected:
  void ConvertInputsToStack(const std::vector<const ir::Value *> &inputs, torch::jit::Stack &stack) const;
  void ConvertStackToOutput(ir::Value *output, torch::jit::Stack &&stack) const;
  void ToMrtTensor(ir::Value *output, torch::jit::IValue ivalue) const;
  bool MatchOpSchema(const std::vector<const ir::Value *> &inputs,
                     const std::shared_ptr<torch::jit::Operator> op) const;
  bool HasSharedStorageWithInput(const ir::Value *output, const ir::Value *input) const;
  std::string GetOpsExpr(const std::vector<const ir::Value *> &inputs) const;
  std::string GetAvailableTorchOps() const;

  void ConvertTensorInputToStack(const ir::Value *value, torch::jit::Stack &stack) const;
  void ConvertDoubleInputToStack(const ir::Value *value, torch::jit::Stack &stack) const;
  void ConvertIntInputToStack(const ir::Value *value, torch::jit::Stack &stack) const;
  void ConvertBoolInputToStack(const ir::Value *value, torch::jit::Stack &stack) const;
  void ConvertStringInputToStack(const ir::Value *value, torch::jit::Stack &stack) const;
  void ConvertTupleInputToStack(const ir::Value *value, torch::jit::Stack &stack) const;
  void ConvertNoneInputToStack(const ir::Value *value, torch::jit::Stack &stack) const;

  void ConvertTupleToStack(const ir::TuplePtr tuple, torch::jit::Stack &stack) const;
  void ConvertTensorTupleToStack(const ir::TuplePtr tuple, torch::jit::Stack &stack) const;
  void ConvertIntTupleToStack(const ir::TuplePtr tuple, torch::jit::Stack &stack) const;
  void ConvertBoolTupleToStack(const ir::TuplePtr tuple, torch::jit::Stack &stack) const;
  void ConvertDoubleTupleToStack(const ir::TuplePtr tuple, torch::jit::Stack &stack) const;
  void ConvertStringTupleToStack(const ir::TuplePtr tuple, torch::jit::Stack &stack) const;
  void ConvertNoneTupleToStack(const ir::TuplePtr tuple, torch::jit::Stack &stack) const;

  std::string qualifiedOpName_;
  torch::jit::Operation operation_ = nullptr;
};
}  // namespace ops
}  // namespace mrt

#endif  // __OPS_TORCH_OP_TORCH_CALL_H__
