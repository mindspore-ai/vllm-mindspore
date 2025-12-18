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

#ifndef __OPS_ASCEND_LOWERED_OP_LINALG_CALL_H__
#define __OPS_ASCEND_LOWERED_OP_LINALG_CALL_H__

#include <memory>
#include <string>
#include <vector>

#include "ops/op_register.h"

namespace mrt {
namespace ops {

/**
 * @brief OpLinalgCall - Standard IR operator for invoking Linalg MLIR compiled operators
 *
 * Similar to OpCustomCall, but:
 * - OpCustomCall: inputs[0] = op_name (string), lookup via CreateCustomOperator(op_name)
 * - OpLinalgCall: inputs[0] = mlir_text (string), creation via LoweredOpHelper::CreateFromMlirText()
 *
 * Input format:
 * - inputs[0]: Linalg MLIR text (String Value) - must contain hacc.entry annotation
 * - inputs[1...N]: Actual input tensors
 *
 * Internal workflow:
 * 1. Init: Extract MLIR text, call LoweredOpHelper::CreateFromMlirText()
 * 2. InferShape/CalcWorkspace/Launch: Delegate to internal AutoLoweredOp
 */
class OpLinalgCall : public Operator {
 public:
  OpLinalgCall() = default;
  ~OpLinalgCall() override = default;

  /**
   * @brief Initialize linalg_call operator
   * @param inputs Input list:
   *   - inputs[0]: Linalg MLIR text (String type Value)
   *   - inputs[1...N]: Actual input tensors
   * @param output Output tensor
   */
  void Init(const std::vector<const ir::Value *> &inputs, const ir::Value *output);

  OpsErrorCode InferShape(const std::vector<const ir::Value *> &input, ir::Value *output) override;

  OpsErrorCode CalcWorkspace(const std::vector<const ir::Value *> &input, const ir::Value *output,
                             size_t *workspaceSize);

  OpsErrorCode Launch(const std::vector<const ir::Value *> &input, void *workspace, size_t workspaceSize,
                      ir::Value *output, void *stream);

 protected:
  std::string mlirText_;                       // Stored MLIR text
  std::unique_ptr<Operator> loweredOp_;        // AutoLoweredOp instance (created via LoweredOpHelper)
  std::vector<const ir::Value *> realInputs_;  // Actual inputs (excluding MLIR text parameter)
};

}  // namespace ops
}  // namespace mrt

#endif  // __OPS_ASCEND_LOWERED_OP_LINALG_CALL_H__
