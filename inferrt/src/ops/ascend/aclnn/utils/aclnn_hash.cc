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

#include "ops/ascend/aclnn/utils/aclnn_hash.h"

namespace mrt {
namespace ops {
constexpr size_t kSizeFive = 5;
void GatherHash(const ir::TensorPtr &tensor) {
  if (tensor == nullptr || tensor->Dtype().value == ir::DataType::Type::Unknown) {
    MemcpyToBuf("None", kSizeFive);
    return;
  }

  // shape
  const auto &shape = tensor->Shape();
  if (!shape.empty()) {
    MemcpyToBuf(shape.data(), tensor->Dim() * sizeof(int64_t));
  }

  // dtype
  auto dtype = tensor->Dtype().value;
  MemcpyToBuf(&dtype, sizeof(int8_t));

  // strides
  const auto &strides = tensor->Strides();
  if (!strides.empty()) {
    MemcpyToBuf(strides.data(), strides.size() * sizeof(int64_t));
  }

  // offset
  auto offset = tensor->StorageOffset();
  MemcpyToBuf(&offset, sizeof(int64_t));
}

}  // namespace ops
}  // namespace mrt
