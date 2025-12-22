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

#ifndef __OPS_CPU_SYMBOLIC_BROADCAST_SHAPE_H__
#define __OPS_CPU_SYMBOLIC_BROADCAST_SHAPE_H__

#include "ops/operator.h"

namespace mrt {
namespace ops {

// Compute broadcasted shape of two tensors and return it as a tuple<int64>.
// This is a host-side metadata op used by dynamic-shape lowering.
class BroadcastShape : public Operator {
 public:
  BroadcastShape() = default;
  ~BroadcastShape() override = default;

  OpsErrorCode InferShape(const std::vector<const ir::Value *> &input, ir::Value *output) override;
  OpsErrorCode Launch(const std::vector<const ir::Value *> &input, void *workspace, size_t workspaceSize,
                      ir::Value *output, void *stream) override;
};

}  // namespace ops
}  // namespace mrt

#endif  // __OPS_CPU_SYMBOLIC_BROADCAST_SHAPE_H__
