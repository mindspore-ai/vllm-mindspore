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

#include "ops/cpu/aten/aten_reshape.h"
#include "ops/cpu/aten/utils/aten_convert.h"
#include "ops/op_register.h"

namespace mrt {
namespace ops {

OpsErrorCode AtenReshape::Launch(const std::vector<const ir::Value *> &input, void *workspace, size_t workspaceSize,
                                 ir::Value *output, void *stream) {
  auto atenInput0 = ToAtenTensor(input[kIndex0]);
  auto atenOutput = ToAtenTensor(output);
  auto outputShape = output->ToTensor()->Shape();
  at::resize_out(atenOutput, atenInput0, outputShape);
  return SUCCESS;
}

MRT_REG_OP(reshape, AtenReshape, CPU);
}  // namespace ops
}  // namespace mrt
