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

#ifndef MOPT_CONVERSION_MRT_TYPE_CONVERTER_H
#define MOPT_CONVERSION_MRT_TYPE_CONVERTER_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mopt/Dialect/Mrt/Mrt.h"

namespace mrt {

// Populate type conversions for converting standard MLIR types to MRT types.
// This adds conversions for RankedTensorType -> mrt::TensorType.
inline void populateMrtTensorTypeConversions(mlir::TypeConverter &converter) {
  converter.addConversion([](mlir::RankedTensorType type) -> mlir::Type {
    return mrt::TensorType::get(type.getContext(), type.getShape(), type.getElementType(), nullptr);
  });
}

// Populate type conversions for standard MLIR scalar types to MRT scalar types.
// This adds conversions for:
//   - IntegerType(1)      -> mrt::BooleanType
//   - IntegerType(8-64)   -> mrt::I64Type
//   - Float16Type/BF16    -> mrt::F64Type
//   - Float32Type         -> mrt::F64Type
//   - Float64Type         -> mrt::F64Type
//   - IndexType           -> mrt::I64Type
inline void populateMrtScalarTypeConversions(mlir::TypeConverter &converter) {
  // Integer types
  converter.addConversion([](mlir::IntegerType type) -> mlir::Type {
    unsigned width = type.getWidth();
    if (width == 1) {
      return mrt::BooleanType::get(type.getContext());
    }
    if (width <= 64) {
      return mrt::I64Type::get(type.getContext());
    }
    return type;  // Keep unsupported widths as-is
  });

  // Float types
  converter.addConversion([](mlir::Float16Type type) -> mlir::Type { return mrt::F64Type::get(type.getContext()); });
  converter.addConversion([](mlir::BFloat16Type type) -> mlir::Type { return mrt::F64Type::get(type.getContext()); });
  converter.addConversion([](mlir::Float32Type type) -> mlir::Type { return mrt::F64Type::get(type.getContext()); });
  converter.addConversion([](mlir::Float64Type type) -> mlir::Type { return mrt::F64Type::get(type.getContext()); });

  // Index type
  converter.addConversion([](mlir::IndexType type) -> mlir::Type { return mrt::I64Type::get(type.getContext()); });
}

// Populate all type conversions for converting standard MLIR types to MRT types.
// This includes both tensor and scalar type conversions.
inline void populateMrtTypeConversions(mlir::TypeConverter &converter) {
  populateMrtTensorTypeConversions(converter);
  populateMrtScalarTypeConversions(converter);
}

// Populate materialization functions for bridging between builtin types and MRT types.
// This uses UnrealizedConversionCastOp for bridging, which should be cleaned up
// later by reconcile-unrealized-casts pass.
inline void populateMrtTypeMaterializations(mlir::TypeConverter &converter) {
  // Target materialization: builtin types -> MRT types
  // Used when creating new ops that expect MRT types
  converter.addTargetMaterialization(
      [](mlir::OpBuilder &builder, mlir::Type toType, mlir::ValueRange inputs,
         mlir::Location loc) -> mlir::Value {
        if (inputs.size() != 1)
          return {};
        return builder.create<mlir::UnrealizedConversionCastOp>(loc, toType, inputs).getResult(0);
      });

  // Source materialization: MRT types -> builtin types
  // Used to maintain compatibility with surrounding code that expects builtin types
  converter.addSourceMaterialization(
      [](mlir::OpBuilder &builder, mlir::Type toType, mlir::ValueRange inputs,
         mlir::Location loc) -> mlir::Value {
        if (inputs.size() != 1)
          return {};
        return builder.create<mlir::UnrealizedConversionCastOp>(loc, toType, inputs).getResult(0);
      });
}
}  // namespace mrt

#endif  // MOPT_CONVERSION_MRT_TYPE_CONVERTER_H
