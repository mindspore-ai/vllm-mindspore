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

#ifndef __OPS_CPU_SYMBOLIC_BINARY_OP_H__
#define __OPS_CPU_SYMBOLIC_BINARY_OP_H__

#include <vector>

#include "ops/operator.h"

namespace mrt {
namespace ops {
class BinaryOp : public Operator {
 public:
  BinaryOp() = default;
  ~BinaryOp() override = default;

  OpsErrorCode InferShape(const std::vector<const ir::Value *> &input, ir::Value *output) override;
  OpsErrorCode Launch(const std::vector<const ir::Value *> &input, void *workspace, size_t workspaceSize,
                      ir::Value *output, void *stream) override;
  bool NeedLaunch() override;
};

#define DefineBinaryOp(op_name)     \
  class op_name : public BinaryOp { \
   public:                          \
    op_name() = default;            \
    ~op_name() override = default;  \
  }

DefineBinaryOp(BinaryAdd);
DefineBinaryOp(BinarySub);
DefineBinaryOp(BinaryMul);
DefineBinaryOp(BinaryDiv);
DefineBinaryOp(BinaryFloorDiv);
}  // namespace ops
}  // namespace mrt

#endif  // __OPS_CPU_SYMBOLIC_BINARY_OP_H__
