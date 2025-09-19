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

#include "ops/op_base/op_shape.h"
#include "ir/value/value.h"
#include "common/logger.h"

namespace mrt {
namespace ops {
OpsErrorCode OpShape::InferShape(const std::vector<const ir::Value *> &input, ir::Value *output) {
  // The output of Op Shape is a tuple, which does not have a shape in the tensor sense.
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
}  // namespace ops
}  // namespace mrt
