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

#ifndef MRT_DIALECT_MRT_CONSTANT_OPS_TD
#define MRT_DIALECT_MRT_CONSTANT_OPS_TD

include "mopt/Dialect/Mrt/MrtTypes.td"

//===----------------------------------------------------------------------===//
// Operations to create values of custom types from attributes
// These operations create initial values that can be modified at runtime.
//===----------------------------------------------------------------------===//

// Create an i64 value from an attribute
// The created value can be modified at runtime through other operations.
def Mrt_CreateI64Op : Mrt_Op<"constant.i64", [Pure]> {
  let summary = "Create an i64 value from an attribute";
  let description = [{
    Creates a value of Mrt_I64Type from an integer attribute.
    The created value can be modified at runtime through other operations.
  }];

  let arguments = (ins I64Attr:$value);
  let results = (outs Mrt_I64Type:$result);

  let assemblyFormat = "$value attr-dict `:` type($result)";
}

// Create a boolean value from an attribute
// The created value can be modified at runtime through other operations.
def Mrt_CreateBooleanOp : Mrt_Op<"constant.boolean", [Pure]> {
  let summary = "Create a boolean value from an attribute";
  let description = [{
    Creates a value of Mrt_BooleanType from a boolean attribute.
    The created value can be modified at runtime through other operations.
  }];

  let arguments = (ins BoolAttr:$value);
  let results = (outs Mrt_BooleanType:$result);

  let assemblyFormat = "$value attr-dict `:` type($result)";
}

// Create a f64 value from an attribute
// The created value can be modified at runtime through other operations.
def Mrt_CreateF64Op : Mrt_Op<"constant.f64", [Pure]> {
  let summary = "Create a f64 value from an attribute";
  let description = [{
    Creates a value of Mrt_F64Type from a double attribute.
    The created value can be modified at runtime through other operations.
  }];

  let arguments = (ins F64Attr:$value);
  let results = (outs Mrt_F64Type:$result);

  let assemblyFormat = "$value attr-dict `:` type($result)";
}

// Create an i64 array value from an attribute
// The created value can be modified at runtime through other operations.
def Mrt_CreateI64ArrayOp : Mrt_Op<"constant.i64_array", [Pure]> {
  let summary = "Create an i64 array value from an attribute";
  let description = [{
    Creates a value of Mrt_I64ArrayType from an integer array attribute.
    The created value can be modified at runtime through other operations.
  }];

  let arguments = (ins I64ArrayAttr:$value);
  let results = (outs Mrt_I64ArrayType:$result);

  let assemblyFormat = "$value attr-dict `:` type($result)";
}

// Create a f64 array value from an attribute
// The created value can be modified at runtime through other operations.
def Mrt_CreateF64ArrayOp : Mrt_Op<"constant.f64_array", [Pure]> {
  let summary = "Create a f64 array value from an attribute";
  let description = [{
    Creates a value of Mrt_F64ArrayType from a double array attribute.
    The created value can be modified at runtime through other operations.
  }];

  let arguments = (ins F64ArrayAttr:$value);
  let results = (outs Mrt_F64ArrayType:$result);

  let assemblyFormat = "$value attr-dict `:` type($result)";
}

// Create a boolean array value from an attribute
// The created value can be modified at runtime through other operations.
def Mrt_CreateBooleanArrayOp : Mrt_Op<"constant.boolean_array", [Pure]> {
  let summary = "Create a boolean array value from an attribute";
  let description = [{
    Creates a value of Mrt_BooleanArrayType from a boolean array attribute.
    The created value can be modified at runtime through other operations.
  }];

  let arguments = (ins BoolArrayAttr:$value);
  let results = (outs Mrt_BooleanArrayType:$result);

  let assemblyFormat = "$value attr-dict `:` type($result)";
}

// Create a string value from an attribute
// The created value can be modified at runtime through other operations.
def Mrt_CreateStringOp : Mrt_Op<"constant.string", [Pure]> {
  let summary = "Create a string value from an attribute";
  let description = [{
    Creates a value of Mrt_StringType from a string attribute.
    The created value can be modified at runtime through other operations.
  }];

  let arguments = (ins StrAttr:$value);
  let results = (outs Mrt_StringType:$result);

  let assemblyFormat = "$value attr-dict `:` type($result)";
}

// Create a dtype value from an attribute
// The created value can be modified at runtime through other operations.
def Mrt_CreateDtypeOp : Mrt_Op<"constant.dtype", [Pure]> {
  let summary = "Create a dtype value from a type attribute";
  let description = [{
    Creates a value of Mrt_DtypeType from a type attribute.
    The type attribute represents an element type (e.g., f32, i64) which is
    stored directly in the Mrt_DtypeType and can be converted to DataType enum when needed.
  }];

  let arguments = (ins TypeAttr:$value);
  let results = (outs Mrt_DtypeType:$result);

  let assemblyFormat = "$value attr-dict `:` type($result)";
}

#endif // MRT_DIALECT_MRT_CONSTANT_OPS_TD
