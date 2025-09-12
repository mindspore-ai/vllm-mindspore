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
OpsErrorCode AtenShape::InferShape(const std::vector<const ir::Value *> &input, ir::Value *output) {
  return OpShape::InferShape(input, output);
}

OpsErrorCode AtenShape::Launch(const std::vector<const ir::Value *> &input, void *workspace, size_t workspaceSize,
                               ir::Value *output, void *stream) {
  auto tensor = input[kIndex0]->ToTensor();
  auto &shape = tensor->Shape();

  std::vector<ir::ValuePtr> shapeValues;
  shapeValues.reserve(shape.size());
  for (auto dim : shape) {
    (void)shapeValues.emplace_back(ir::MakeIntrusive<ir::Value>(dim));
  }

  auto tuple_data = ir::MakeIntrusive<ir::Tuple>(shapeValues);
  *output = ir::Value(tuple_data);

  return SUCCESS;
}

MRT_REG_OP(shape, AtenShape, CPU);
}  // namespace ops
}  // namespace mrt
