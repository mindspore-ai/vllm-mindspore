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

#include "ops/op_base/op_eval_symbolic_expr.h"

#include "ir/value/value.h"
#include "ir/symbolic/symbolic.h"
#include "common/logger.h"

namespace mrt {
namespace ops {

// Inputs: [operand_0, operand_1, ..., operand_N-1, symVars]
// Output: symExpr
// Set the value of symVars with operands, such that symExpr in output can be evaluated later.
OpsErrorCode OpEvalSymbolicExpr::InferShape(const std::vector<const ir::Value *> &input, ir::Value *output) {
  if (input.size() < 1) {
    LOG_EXCEPTION << "OpEvalSymbolicExpr: input size must be at least 1";
  }

  size_t numOperands = input.size() - 1;
  auto symVars = input[numOperands]->ToTuple();

  if (symVars->Size() != numOperands) {
    LOG_EXCEPTION << "OpEvalSymbolicExpr: symVars size must be equal to numOperands";
  }

  for (size_t i = 0; i < numOperands; ++i) {
    auto symVarExpr = (*symVars)[i]->ToSymbol();
    auto symVar = static_cast<ir::SymbolicVar *>(symVarExpr.get());
    symVar->SetValue(input[i]->ToInt());
  }

  return SUCCESS;
}

OpsErrorCode OpEvalSymbolicExpr::Launch(const std::vector<const ir::Value *> &input, void *workspace,
                                        size_t workspaceSize, ir::Value *output, void *stream) {
  return SUCCESS;
}

}  // namespace ops
}  // namespace mrt
