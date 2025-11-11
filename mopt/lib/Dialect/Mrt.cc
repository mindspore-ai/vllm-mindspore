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

#include "mopt/Dialect/Mrt/Mrt.h"
#include "mlir/IR/Types.h"

using mlir::LogicalResult;
using mlir::RankedTensorType;
using mlir::success;

namespace mrt {
LogicalResult ReshapeOp::verify() {
  auto shapeTy = getShape().getType().dyn_cast<RankedTensorType>();
  if (!shapeTy) {
    return emitOpError("expects shape to be a ranked tensor of i64");
  }
  if (shapeTy.getRank() != 1) {
    return emitOpError("expects shape tensor to be rank-1");
  }
  if (!shapeTy.getElementType().isInteger(64)) {
    return emitOpError("expects shape tensor element type to be i64");
  }
  return success();
}
}  // namespace mrt
