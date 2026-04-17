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

#include "ops/ascend/aclnn/utils/view_utils.h"

#include "common/logger.h"
namespace mrt {
namespace ops {
std::vector<int64_t> CalculateStrides(const std::vector<int64_t> &shape) {
  if (shape.empty()) {
    return {};
  }

  std::vector<int64_t> ret(shape.size(), 1);
  int64_t strides = 1;
  for (size_t i = shape.size() - 1; i > 0; --i) {
    strides *= shape[i];
    ret[i - 1] = strides;
  }
  return ret;
}

int64_t DynamicDimWrap(int64_t dim, int64_t dimPostExpr, bool wrapScalar) {
  if (dimPostExpr * -1 <= dim && dim < dimPostExpr) {
    if (dim < 0) {
      return dim + dimPostExpr;
    }
    return dim;
  }
  if (dimPostExpr == 0) {
    if (!wrapScalar) {
      LOG_EXCEPTION << "dim value specified as " << dim << ", but tensor has no dimensions";
    }
    return DynamicDimWrap(dim, 1, false);
  }
  LOG_EXCEPTION << "Dimension out of range (expected to be in range of [" << -dimPostExpr << ", " << dimPostExpr
                << "), but got " << dim << ")";
  return -1;
}

std::vector<int64_t> GetTensorStrides(const ir::TensorPtr &tensorPtr) {
  const auto &strides = tensorPtr->Strides();
  if (strides.empty()) {
    return CalculateStrides(tensorPtr->Shape());
  }
  return strides;
}

void UpdateTensorViewInfo(const ir::TensorPtr &inputTensorPtr, const ir::TensorPtr &outputTensorPtr,
                          const std::vector<int64_t> &newShape, const std::vector<int64_t> &newStrides,
                          size_t newStorageOffset) {
  outputTensorPtr->SetShape(newShape);
  outputTensorPtr->SetStrides(newStrides);
  outputTensorPtr->SetStorageOffset(newStorageOffset);
  outputTensorPtr->SetStorageShape(inputTensorPtr->StorageShape());
  outputTensorPtr->SetFormat(inputTensorPtr->Format());
}

std::vector<std::pair<uint32_t, uint32_t>> GenerateOutputInputRefPair(const ir::Value *output) {
  std::vector<std::pair<uint32_t, uint32_t>> result;

  if (output->IsTuple()) {
    const auto numOutput = output->ToTuple()->Size();
    result.reserve(numOutput);
    for (uint32_t i = 0; i < numOutput; ++i) {
      result.emplace_back(i, 0);
    }
  } else if (output->IsTensor()) {
    result.emplace_back(0, 0);
  } else {
    LOG_EXCEPTION << "Output is not a tensor or tuple";
  }

  return result;
}
}  // namespace ops
}  // namespace mrt
