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

#include "mopt/Dialect/Mrt/MrtValueBuilder.h"

#include "mopt/Dialect/Mrt/Mrt.h"

namespace mrt {

mlir::Value MrtValueBuilder::createI64Array(mlir::Location loc, llvm::ArrayRef<int64_t> values) {
  auto arrayAttr = builder_.getI64ArrayAttr(values);
  auto arrayType = I64ArrayType::get(builder_.getContext());
  return builder_.create<CreateI64ArrayOp>(loc, arrayType, arrayAttr);
}

mlir::Value MrtValueBuilder::createI64(mlir::Location loc, int64_t value) {
  auto intAttr = builder_.getI64IntegerAttr(value);
  auto intType = I64Type::get(builder_.getContext());
  return builder_.create<CreateI64Op>(loc, intType, intAttr);
}

mlir::Value MrtValueBuilder::createBoolean(mlir::Location loc, bool value) {
  auto boolAttr = builder_.getBoolAttr(value);
  auto boolType = BooleanType::get(builder_.getContext());
  return builder_.create<CreateBooleanOp>(loc, boolType, boolAttr);
}

mlir::Value MrtValueBuilder::createF64(mlir::Location loc, double value) {
  auto floatAttr = builder_.getF64FloatAttr(value);
  auto f64Type = F64Type::get(builder_.getContext());
  return builder_.create<CreateF64Op>(loc, f64Type, floatAttr);
}

mlir::Value MrtValueBuilder::createDtype(mlir::Location loc, mlir::Type elemType) {
  auto typeAttr = mlir::TypeAttr::get(elemType);
  auto dtypeType = DtypeType::get(builder_.getContext(), elemType);
  return builder_.create<CreateDtypeOp>(loc, dtypeType, typeAttr);
}

}  // namespace mrt
