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

#include <vector>

#include "common/logger.h"
#include "ir/value/value.h"

#include "ops/cpu/aten/aten_shape.h"
#include "ops/op_register.h"

namespace mrt {
namespace ops {

OpsErrorCode AtenShape::Launch(const std::vector<const ir::Value *> &input, void *workspace, size_t workspaceSize,
                               ir::Value *output, void *stream) {
  // The output of Shape is a tuple, which does not have a shape in the tensor sense.
  // The tuple will be constructed in the InferShape method, only need input tensor shape information.
  // Here we just skip launch.

  return SUCCESS;
}

MRT_REG_OP(shape, AtenShape, CPU);
}  // namespace ops
}  // namespace mrt
