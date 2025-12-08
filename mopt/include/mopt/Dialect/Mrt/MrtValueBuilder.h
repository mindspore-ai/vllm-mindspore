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

#ifndef MOPT_DIALECT_MRT_MRT_VALUE_BUILDER_H
#define MOPT_DIALECT_MRT_MRT_VALUE_BUILDER_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/ArrayRef.h"

namespace mrt {

// Helper class for creating MRT constant values.
class MrtValueBuilder {
 public:
  explicit MrtValueBuilder(mlir::OpBuilder &builder) : builder_(builder) {}

  // Create an i64 array value from shape values
  mlir::Value createI64Array(mlir::Location loc, llvm::ArrayRef<int64_t> values);

  // Create an i64 value
  mlir::Value createI64(mlir::Location loc, int64_t value);

  // Create a boolean value
  mlir::Value createBoolean(mlir::Location loc, bool value);

  // Create a f64 value
  mlir::Value createF64(mlir::Location loc, double value);

  // Create a dtype value from element type
  mlir::Value createDtype(mlir::Location loc, mlir::Type elemType);

 private:
  mlir::OpBuilder &builder_;
};

}  // namespace mrt

#endif  // MOPT_DIALECT_MRT_MRT_VALUE_BUILDER_H
