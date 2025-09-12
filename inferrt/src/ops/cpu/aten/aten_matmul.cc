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

#include "ops/cpu/aten/aten_matmul.h"
#include "ops/cpu/aten/utils/aten_convert.h"
#include "ops/op_register.h"

namespace mrt {
namespace ops {
OpsErrorCode AtenMatMul::InferShape(const std::vector<const ir::Value *> &input, ir::Value *output) {
  if (input.size() != kInputSize2) {
    LOG_ERROR << "Expect input size is 2 for AtenMatMul, but got: " << input.size();
    return INVALID_INPUT_NUM;
  }
  auto &input0Shape = input[kIndex0]->ToTensor()->Shape();
  auto &input1Shape = input[kIndex1]->ToTensor()->Shape();
  std::vector<int64_t> outputShape = {input0Shape[0], input1Shape[1]};
  CHECK_IF_NULL(output);
  auto outputTensor = output->ToTensor();
  CHECK_IF_NULL(outputTensor);
  outputTensor->SetShape(outputShape);
  auto outputDtype = input[kIndex0]->ToTensor()->Dtype();
  outputTensor->SetDtype(outputDtype);
  outputTensor->ResizeStorage();
  return SUCCESS;
}

OpsErrorCode AtenMatMul::Launch(const std::vector<const ir::Value *> &input, void *workspace, size_t workspaceSize,
                                ir::Value *output, void *stream) {
  auto atenInput0 = ToAtenTensor(input[kIndex0]);
  auto atenInput1 = ToAtenTensor(input[kIndex1]);
  auto atenOutput = ToAtenTensor(output);
  at::matmul_out(atenOutput, atenInput0, atenInput1);
  return SUCCESS;
}

MRT_REG_OP(matmul, AtenMatMul, CPU);
}  // namespace ops
}  // namespace mrt
