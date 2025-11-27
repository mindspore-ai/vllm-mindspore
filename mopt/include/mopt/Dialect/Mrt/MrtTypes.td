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

#ifndef MRT_DIALECT_MRT_TYPES_TD
#define MRT_DIALECT_MRT_TYPES_TD

include "mopt/Dialect/Mrt/MrtDialect.td"
include "mlir/IR/AttrTypeBase.td"

//===----------------------------------------------------------------------===//
// Supported argument/result types for operations:
//   Tensor          --> Mrt_TensorType
//   Tensor List     --> MrtTensorList
//   Integer         --> Mrt_I64Type (replaces I64Attr)
//   Integer Array   --> Mrt_I64ArrayType (replaces I64ArrayAttr, for axis, strides, shape, etc.)
//   Boolean         --> Mrt_BooleanType (replaces BoolAttr)
//   Boolean Array   --> Mrt_BooleanArrayType (replaces BoolArrayAttr)
//   Float           --> Mrt_F32Type or Mrt_F64Type (replaces F32Attr/F64Attr)
//   Float Array     --> Mrt_F32ArrayType or Mrt_F64ArrayType (replaces F32ArrayAttr/F64ArrayAttr)
//   String          --> Mrt_StringType (replaces StrAttr)
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Device Attribute
//===----------------------------------------------------------------------===//

def Mrt_DeviceAttr : AttrDef<Mrt_Dialect, "Device"> {
  let mnemonic = "device";
  let summary = "Device attribute for MRT dialect";
  let description = [{
    Represents a device with type (e.g., "cpu", "npu") and index (signed integer).
    The index is a signed integer indicating the device index.
  }];

  let parameters = (ins
    "mlir::StringAttr":$device_type,
    "mlir::IntegerAttr":$index
  );

  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// Tensor Types
//===----------------------------------------------------------------------===//

def Mrt_TensorType : Mrt_Type<"Tensor", "tensor"> {
  let summary = "Tensor type for MRT dialect";
  let description = [{
    Represents a multi-dimensional array with shape, element type, and optional device information.
    Device information includes device type (e.g., "cpu", "npu") and device index.
    Format: mrt.tensor<elementTypexshape, device=...> (e.g., mrt.tensor<f32x2x3, device=<npu, 0>>)
  }];

  let parameters = (ins
    "mlir::Type":$elementType,
    "mlir::DenseI64ArrayAttr":$shape,
    OptionalParameter<"mrt::DeviceAttr">:$device
  );

  let hasCustomAssemblyFormat = 1;
}


// Type constraint that accepts both RankedTensorType and Mrt_TensorType
// This allows operations to accept either standard MLIR RankedTensorType
// or MRT dialect's TensorType, providing flexibility during conversion.
def MrtAnyTensor : AnyTypeOf<[AnyRankedTensor, Mrt_TensorType]>;

// Tensor list type (uses MrtAnyTensor to accept both RankedTensorType and Mrt_TensorType)
def MrtTensorList : Variadic<MrtAnyTensor>;

//===----------------------------------------------------------------------===//
// Scalar Types
//===----------------------------------------------------------------------===//

// 32-bit floating point scalar type (replaces F32Attr)
// This type wraps a 32-bit floating point value and is distinct from tensor types.
// Use this type as an operand to pass scalar float values to operations.
def Mrt_F32Type : Mrt_Type<"F32", "f32"> {
  let summary = "32-bit floating point scalar type for MRT dialect";
  let description = [{
    Represents a 32-bit floating point scalar value. This type is used as an operand
    to replace F32Attr, allowing all parameters to be passed as operands.
    This type is distinct from tensor types and can be used to pass scalar
    floating point values to operations.
  }];

  let builders = [
    TypeBuilderWithInferredContext<(ins), [{
      return Base::get(context);
    }]>
  ];

  let assemblyFormat = "";
}

// 64-bit floating point scalar type (replaces F64Attr)
// This type wraps a 64-bit floating point value and is distinct from tensor types.
// Use this type as an operand to pass scalar double values to operations.
def Mrt_F64Type : Mrt_Type<"F64", "f64"> {
  let summary = "64-bit floating point scalar type for MRT dialect";
  let description = [{
    Represents a 64-bit floating point scalar value. This type is used as an operand
    to replace F64Attr, allowing all parameters to be passed as operands.
    This type is distinct from tensor types and can be used to pass scalar
    floating point values to operations.
  }];

  let builders = [
    TypeBuilderWithInferredContext<(ins), [{
      return Base::get(context);
    }]>
  ];

  let assemblyFormat = "";
}

// 64-bit integer scalar type (replaces I64Attr)
// This type wraps a 64-bit integer value and is distinct from tensor types.
// Use this type as an operand to pass scalar integer values to operations.
def Mrt_I64Type : Mrt_Type<"I64", "i64"> {
  let summary = "64-bit integer scalar type for MRT dialect";
  let description = [{
    Represents a 64-bit integer scalar value. This type is used as an operand
    to replace I64Attr, allowing all parameters to be passed as operands.
    This type is distinct from tensor types and can be used to pass scalar
    integer values to operations.
  }];

  let builders = [
    TypeBuilderWithInferredContext<(ins), [{
      return Base::get(context);
    }]>
  ];

  let assemblyFormat = "";
}

// Boolean scalar type (replaces BoolAttr)
// This type wraps a boolean value and is distinct from tensor types.
// Use this type as an operand to pass scalar boolean values to operations.
def Mrt_BooleanType : Mrt_Type<"Boolean", "boolean"> {
  let summary = "Boolean scalar type for MRT dialect";
  let description = [{
    Represents a boolean scalar value. This type is used as an operand
    to replace BoolAttr, allowing all parameters to be passed as operands.
    This type is distinct from tensor types and can be used to pass scalar
    boolean values to operations.
  }];

  let builders = [
    TypeBuilderWithInferredContext<(ins), [{
      return Base::get(context);
    }]>
  ];

  let assemblyFormat = "";
}

// String type (replaces StrAttr)
// This type wraps a string value and is distinct from tensor types.
// Use this type as an operand to pass string values to operations.
def Mrt_StringType : Mrt_Type<"String", "string"> {
  let summary = "String type for MRT dialect";
  let description = [{
    Represents a string value. This type is used as an operand
    to replace StrAttr, allowing all parameters to be passed as operands.
    This type is distinct from tensor types and can be used to pass string
    values to operations.
  }];

  let builders = [
    TypeBuilderWithInferredContext<(ins), [{
      return Base::get(context);
    }]>
  ];

  let assemblyFormat = "";
}

//===----------------------------------------------------------------------===//
// Array Types
//===----------------------------------------------------------------------===//

// 64-bit integer array type (replaces I64ArrayAttr)
// This type represents a general-purpose array of int64 values and is distinct from tensor types.
// Use this type as an operand to pass integer arrays for various purposes such as
// axis, strides, padding, dilation, kernel_size, split_sizes, shape, etc.
def Mrt_I64ArrayType : Mrt_Type<"I64Array", "i64_array"> {
  let summary = "64-bit integer array type for MRT dialect";
  let description = [{
    Represents an array of int64 values for general-purpose use cases such as
    axis indices, strides, padding, dilation, kernel sizes, split sizes, shape, etc.
    This type is used as an operand to replace I64ArrayAttr, allowing all
    parameters to be passed as operands. This type is distinct from tensor
    types and can be used to pass integer arrays to operations.
  }];

  let builders = [
    TypeBuilderWithInferredContext<(ins), [{
      return Base::get(context);
    }]>
  ];

  let assemblyFormat = "";
}

// 32-bit floating point array type (replaces F32ArrayAttr)
// This type represents an array of float32 values and is distinct from tensor types.
// Use this type as an operand to pass float arrays to operations.
def Mrt_F32ArrayType : Mrt_Type<"F32Array", "f32_array"> {
  let summary = "32-bit floating point array type for MRT dialect";
  let description = [{
    Represents an array of float32 values. This type is used as an operand
    to replace F32ArrayAttr, allowing all parameters to be passed as operands.
    This type is distinct from tensor types and can be used to pass float
    arrays to operations.
  }];

  let builders = [
    TypeBuilderWithInferredContext<(ins), [{
      return Base::get(context);
    }]>
  ];

  let assemblyFormat = "";
}

// 64-bit floating point array type (replaces F64ArrayAttr)
// This type represents an array of float64 values and is distinct from tensor types.
// Use this type as an operand to pass double arrays to operations.
def Mrt_F64ArrayType : Mrt_Type<"F64Array", "f64_array"> {
  let summary = "64-bit floating point array type for MRT dialect";
  let description = [{
    Represents an array of float64 values. This type is used as an operand
    to replace F64ArrayAttr, allowing all parameters to be passed as operands.
    This type is distinct from tensor types and can be used to pass double
    arrays to operations.
  }];

  let builders = [
    TypeBuilderWithInferredContext<(ins), [{
      return Base::get(context);
    }]>
  ];

  let assemblyFormat = "";
}

// Boolean array type (replaces BoolArrayAttr)
// This type represents an array of boolean values and is distinct from tensor types.
// Use this type as an operand to pass boolean arrays to operations.
def Mrt_BooleanArrayType : Mrt_Type<"BooleanArray", "boolean_array"> {
  let summary = "Boolean array type for MRT dialect";
  let description = [{
    Represents an array of boolean values. This type is used as an operand
    to replace BoolArrayAttr, allowing all parameters to be passed as operands.
    This type is distinct from tensor types and can be used to pass boolean
    arrays to operations.
  }];

  let builders = [
    TypeBuilderWithInferredContext<(ins), [{
      return Base::get(context);
    }]>
  ];

  let assemblyFormat = "";
}

// Dtype type (stores mlir::Type directly)
// This type wraps a MLIR element type and is distinct from tensor types.
// Use this type as an operand to pass data type values to operations like Cast.
def Mrt_DtypeType : Mrt_Type<"Dtype", "dtype"> {
  let summary = "Dtype type for MRT dialect";
  let description = [{
    Represents a data type value by storing a MLIR element type directly.
    This type is used as an operand to pass data type values to operations like Cast.
    The element type (e.g., f32, i64) is stored directly, which can be converted
    to DataType enum when needed.
  }];

  let parameters = (ins
    "mlir::Type":$elementType
  );

  let assemblyFormat = [{
    `<` $elementType `>`
  }];
}

// Scalar type (stores mlir::Type for element type)
// This type represents a single-element scalar value with a specific data type.
// Use this type as an operand to pass scalar values of different types to operations.
// For example, it can be used for the alpha parameter in Add operation.
def Mrt_ScalarType : Mrt_Type<"Scalar", "scalar"> {
  let summary = "Scalar type for MRT dialect";
  let description = [{
    Represents a single-element scalar value with a specific data type.
    This type is used as an operand to pass scalar values of different types
    (e.g., f32, f64, i64) to operations. The element type specifies the data type
    of the scalar value. This type is distinct from tensor types and can be used
    to pass scalar values like alpha parameter in Add operation.
  }];

  let parameters = (ins
    "mlir::Type":$elementType
  );

  let assemblyFormat = [{
    `<` $elementType `>`
  }];
}

#endif // MRT_DIALECT_MRT_TYPES_TD
