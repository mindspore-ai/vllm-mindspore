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

#include "ops/cpu/symbolic/size.h"
#include "ops/op_register.h"
#include "ops/utils/op_constants.h"
#include "common/logger.h"
#include "ir/common/dtype.h"

namespace mrt {
namespace ops {
OpsErrorCode Size::InferShape(const std::vector<const ir::Value *> &input, ir::Value *output) {
  const auto &shape = input[kIndex0]->ToTensor()->Shape();
  int64_t dim = input[kIndex1]->ToInt();

  // Support negative dim indexing: -n <= dim < n
  int64_t ndim = static_cast<int64_t>(shape.size());
  CHECK_IF_FAIL(dim >= -ndim && dim < ndim);
  if (dim < 0) {
    dim += ndim;
  }

  *output = ir::Value(shape[dim]);
  return SUCCESS;
}

OpsErrorCode Size::Launch(const std::vector<const ir::Value *> &input, void *workspace, size_t workspaceSize,
                          ir::Value *output, void *stream) {
  return SUCCESS;
}

MRT_REG_OP(size, Size, CPU);
}  // namespace ops
}  // namespace mrt
