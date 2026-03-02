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

#include <vector>

#include "ops/cpu/symbolic/binary_op.h"
#include "ops/op_register.h"

namespace mrt {
namespace ops {

OpsErrorCode BinaryOp::InferShape(const std::vector<const ir::Value *> &input, ir::Value *output) {
  Operator::InferShape(input, output);
  if (!output->IsSymbol() && !output->IsInt() && !output->IsDouble()) {
    LOG_EXCEPTION << "BinaryOp: output must be symbol";
  }
  return SUCCESS;
}

OpsErrorCode BinaryOp::Launch(const std::vector<const ir::Value *> &input, void *workspace, size_t workspaceSize,
                              ir::Value *output, void *stream) {
  return SUCCESS;
}

bool BinaryOp::NeedLaunch() { return false; }

MRT_REG_OP(add_scalar, BinaryAdd, CPU);
MRT_REG_OP(sub_scalar, BinarySub, CPU);
MRT_REG_OP(mul_scalar, BinaryMul, CPU);
MRT_REG_OP(div_scalar, BinaryDiv, CPU);
MRT_REG_OP(div_mod_scalar, BinaryFloorDiv, CPU);
}  // namespace ops
}  // namespace mrt
