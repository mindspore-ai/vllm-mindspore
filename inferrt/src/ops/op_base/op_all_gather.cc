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

#include "ops/op_base/op_all_gather.h"

namespace mrt {
namespace ops {
OpsErrorCode OpAllGather::InferShape(const std::vector<const ir::Value *> &input, ir::Value *output) {
  LOG_OUT << "HcclAllGather InferShape";

  auto &input0Shape = input[kIndex0]->ToTensor()->Shape();
  auto rank_size = input[kIndex1]->ToInt();

  auto outputShape = input0Shape;
  outputShape[0] *= rank_size;

  auto outputTensor = output->ToTensor();
  CHECK_IF_NULL(outputTensor);
  outputTensor->SetShape(outputShape);
  auto outputDtype = input[kIndex0]->ToTensor()->Dtype();
  outputTensor->SetDtype(outputDtype);
  outputTensor->Resize();

  return SUCCESS;
}
}  // namespace ops
}  // namespace mrt
