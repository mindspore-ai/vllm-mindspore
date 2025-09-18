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

#include "ops/op_base/op_reshape.h"
#include "ir/value/value.h"
#include "common/logger.h"

namespace mrt {
namespace ops {
OpsErrorCode OpReshape::InferShape(const std::vector<const ir::Value *> &input, ir::Value *output) {
  if (input.size() != kInputSize2) {
    LOG_ERROR << "Expect input size is 2, but got: " << input.size();
    return INVALID_INPUT_NUM;
  }
  if (!input[kIndex1]->IsTuple()) {
    LOG_ERROR << "Input types are invalid, expect tuple on second input.";
    return INVALID_PARAM;
  }
  auto &shapeTuple = input[kIndex1]->ToTuple();

  auto &outputTensor = output->ToTensor();
  auto &outputShape = outputTensor->Shape();
  outputShape.clear();

  bool hasNegativeDim = false;
  int64_t knownDimProduct = 1;
  for (auto &dimValue : *shapeTuple) {
    int64_t dim = dimValue->ToInt();
    if (dim < 0) {
      if (hasNegativeDim) {
        LOG_EXCEPTION << "Input shape tuple has more than one negative dimension.";
      }
      hasNegativeDim = true;
    } else {
      knownDimProduct *= dim;
    }
    (void)outputShape.emplace_back(dim);
  }

  if (hasNegativeDim) {
    int64_t inputNumel = input[kIndex0]->ToTensor()->Numel();
    if (inputNumel % knownDimProduct != 0) {
      LOG_EXCEPTION << "Input tensor size is invalid for inferring the negative dimension.";
    }
    for (auto &dim : outputShape) {
      if (dim < 0) {
        dim = inputNumel / knownDimProduct;
        break;
      }
    }
  }

  outputTensor->Resize();
  return SUCCESS;
}
}  // namespace ops
}  // namespace mrt
