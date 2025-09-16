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

#include "ops/op_base/op_matmul.h"

namespace mrt {
namespace ops {
OpsErrorCode OpMatMul::InferShape(const std::vector<const ir::Value *> &input, ir::Value *output) {
  if (input.size() != kInputSize2) {
    LOG_ERROR << "Expect input size is 2 for AtenMatMul, but got: " << input.size();
    return INVALID_INPUT_NUM;
  }
  auto &input0Shape = input[kIndex0]->ToTensor()->Shape();
  auto &input1Shape = input[kIndex1]->ToTensor()->Shape();
  auto &outputTensor = output->ToTensor();
  auto &outputShape = outputTensor->Shape();
  outputShape.resize(2);
  outputShape[0] = input0Shape[0];
  outputShape[1] = input1Shape[1];
  outputTensor->Resize();
  return SUCCESS;
}
}  // namespace ops
}  // namespace mrt
