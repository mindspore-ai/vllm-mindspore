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

#ifndef MRT_DIALECT_MRT_OPS_TD
#define MRT_DIALECT_MRT_OPS_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"

def Mrt_Dialect : Dialect {
  let name = "mrt";
  let cppNamespace = "::mrt";
  
  let description = [{
    The MRT (Machine Learning Runtime) dialect provides a set of operations and 
    types for representing machine learning computations. This dialect is designed 
    to be a high-level MLIR dialect that can be lowered to various targets.
  }];
  
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
}

class Mrt_Type<string name, string typeMnemonic> : TypeDef<Mrt_Dialect, name> {
  let mnemonic = typeMnemonic;
}

//===----------------------------------------------------------------------===//
// Supported argument/result types for operations:
//   Tensor          --> AnyRankedTensor
//   Optional Tensor --> MrtOptTensor
//   Tensor List     --> MrtTensorList
//   Integer         --> Mrt_I64Type (replaces I64Attr)
//   Integer Array   --> Mrt_I64ArrayType (replaces I64ArrayAttr, for axis, strides, shape, etc.)
//   Boolean         --> Mrt_BooleanType (replaces BoolAttr)
//   Boolean Array   --> Mrt_BooleanArrayType (replaces BoolArrayAttr)
//   Float           --> Mrt_F32Type or Mrt_F64Type (replaces F32Attr/F64Attr)
//   Float Array     --> Mrt_F32ArrayType or Mrt_F64ArrayType (replaces F32ArrayAttr/F64ArrayAttr)
//   String          --> Mrt_StringType (replaces StrAttr)
//===----------------------------------------------------------------------===//

// Optional tensor type
def MrtOptTensor : Optional<AnyRankedTensor>;

// Tensor list type
def MrtTensorList : Variadic<AnyRankedTensor>;

// Base class for all MRT dialect operations.
class Mrt_Op<string mnemonic, list<Trait> traits = []> : 
    Op<Mrt_Dialect, mnemonic, traits> {
  // Whether generate aclnn launch code.
  // When set to '1', the aclnn launch code will be auto generated.
  bit generated = 0;
}

def Mrt_TensorType : Mrt_Type<"Tensor", "tensor"> {
  let summary = "Tensor type for MRT dialect";
  let description = [{
    Represents a multi-dimensional array with optional shape information.
  }];
  
  let parameters = (ins
    OptionalParameter<"mlir::Type">:$elementType,
    OptionalParameter<"mlir::DenseI64ArrayAttr">:$shape
  );
  
  let assemblyFormat = [{
    `<` $elementType (`x` $shape^)? `>`
  }];
}

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

// Create a f32 value from an attribute
// The created value can be modified at runtime through other operations.
def Mrt_CreateF32Op : Mrt_Op<"constant.f32", [Pure]> {
  let summary = "Create a f32 value from an attribute";
  let description = [{
    Creates a value of Mrt_F32Type from a float attribute.
    The created value can be modified at runtime through other operations.
  }];
  
  let arguments = (ins F32Attr:$value);
  let results = (outs Mrt_F32Type:$result);
  
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

// Create a f32 array value from an attribute
// The created value can be modified at runtime through other operations.
def Mrt_CreateF32ArrayOp : Mrt_Op<"constant.f32_array", [Pure]> {
  let summary = "Create a f32 array value from an attribute";
  let description = [{
    Creates a value of Mrt_F32ArrayType from a float array attribute.
    The created value can be modified at runtime through other operations.
  }];
  
  let arguments = (ins F32ArrayAttr:$value);
  let results = (outs Mrt_F32ArrayType:$result);
  
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

//===----------------------------------------------------------------------===//
// MRT Operations
//===----------------------------------------------------------------------===//

def Mrt_ConvOp : Mrt_Op<"conv", [Pure]> {
  let summary = "2D convolution operation";
  let description = [{
    Performs a 2D convolution operation on input tensors.
  }];
  
  let arguments = (ins
    AnyRankedTensor:$input,
    AnyRankedTensor:$kernel,
    MrtOptTensor:$bias,
    Mrt_I64ArrayType:$strides,
    Mrt_I64ArrayType:$padding,
    Mrt_I64ArrayType:$dilation,
    Mrt_BooleanType:$has_bias
  );
  
  let results = (outs AnyRankedTensor:$result);
  
  let assemblyFormat = [{
    $input `,` $kernel (`,` $bias^)? `,` $strides `,` $padding `,` $dilation `,` $has_bias
    attr-dict `:` functional-type(operands, results)
  }];
}

def Mrt_MatmulOp : Mrt_Op<"matmul", [Pure]> {
  let summary = "Matrix multiplication operation";
  let description = [{
    Performs matrix multiplication on input tensors.
  }];
  
  let arguments = (ins
    AnyRankedTensor:$lhs,
    AnyRankedTensor:$rhs
  );
  
  let results = (outs AnyRankedTensor:$result);
  
  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` functional-type(operands, results)
  }];
}

def Mrt_ReluOp : Mrt_Op<"relu", [Pure]> {
  let summary = "ReLU activation function";
  let description = [{
    Applies the ReLU (Rectified Linear Unit) activation function.
  }];
  
  let arguments = (ins AnyRankedTensor:$input);
  let results = (outs AnyRankedTensor:$result);
  
  let assemblyFormat = [{
    $input attr-dict `:` functional-type(operands, results)
  }];
}

def Mrt_SigmoidOp : Mrt_Op<"sigmoid", [Pure]> {
  let summary = "Sigmoid activation function";
  let description = [{
    Applies the sigmoid activation function.
  }];
  
  let arguments = (ins AnyRankedTensor:$input);
  let results = (outs AnyRankedTensor:$result);
  
  let assemblyFormat = [{
    $input attr-dict `:` functional-type(operands, results)
  }];
}

def Mrt_SoftmaxOp : Mrt_Op<"softmax", [Pure]> {
  let summary = "Softmax activation function";
  let description = [{
    Applies the softmax activation function along the specified axis.
  }];
  
  let arguments = (ins
    AnyRankedTensor:$input,
    Mrt_I64Type:$axis
  );
  let results = (outs AnyRankedTensor:$result);
  
  let assemblyFormat = [{
    $input `,` $axis attr-dict `:` functional-type(operands, results)
  }];
}

def Mrt_BatchNormOp : Mrt_Op<"batch_norm", [Pure]> {
  let summary = "Batch normalization operation";
  let description = [{
    Performs batch normalization on input tensors.
  }];
  
  let arguments = (ins
    AnyRankedTensor:$input,
    AnyRankedTensor:$scale,
    AnyRankedTensor:$offset,
    AnyRankedTensor:$mean,
    AnyRankedTensor:$variance,
    Mrt_F32Type:$epsilon,
    Mrt_BooleanType:$is_training
  );
  
  let results = (outs AnyRankedTensor:$result);
  
  let assemblyFormat = [{
    $input `,` $scale `,` $offset `,` $mean `,` $variance `,` $epsilon `,` $is_training
    attr-dict `:` functional-type(operands, results)
  }];
}

def Mrt_PoolOp : Mrt_Op<"pool", [Pure]> {
  let summary = "Pooling operation";
  let description = [{
    Performs pooling operations (max, average, etc.) on input tensors.
  }];
  
  let arguments = (ins
    AnyRankedTensor:$input,
    Mrt_I64ArrayType:$kernel_size,
    Mrt_I64ArrayType:$strides,
    Mrt_I64ArrayType:$padding,
    Mrt_StringType:$pool_type
  );
  
  let results = (outs AnyRankedTensor:$result);
  
  let assemblyFormat = [{
    $input `,` $kernel_size `,` $strides `,` $padding `,` $pool_type
    attr-dict `:` functional-type(operands, results)
  }];
}

def Mrt_ReshapeOp : Mrt_Op<"reshape", [Pure]> {
  let summary = "Reshape operation";
  let description = [{
    Reshapes input tensor to the specified shape.
    Takes two operands: input tensor and shape.
  }];
  
  let arguments = (ins
    AnyRankedTensor:$input,
    Mrt_I64ArrayType:$shape
  );
  
  let results = (outs AnyRankedTensor:$result);
  
  let assemblyFormat = [{
    $input `,` $shape attr-dict `:` functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

def Mrt_ConcatOp : Mrt_Op<"concat", [Pure]> {
  let summary = "Concatenation operation";
  let description = [{
    Concatenates tensors along the specified axis.
  }];
  
  let arguments = (ins
    MrtTensorList:$inputs,
    Mrt_I64Type:$axis
  );
  
  let results = (outs AnyRankedTensor:$result);
  
  let assemblyFormat = [{
    $inputs `,` $axis attr-dict `:` functional-type(operands, results)
  }];
}

def Mrt_SplitOp : Mrt_Op<"split", [Pure]> {
  let summary = "Split operation";
  let description = [{
    Splits tensor along the specified axis into multiple outputs.
  }];
  
  let arguments = (ins
    AnyRankedTensor:$input,
    Mrt_I64Type:$axis,
    Mrt_I64ArrayType:$split_sizes
  );
  
  let results = (outs MrtTensorList:$results);
  
  let assemblyFormat = [{
    $input `,` $axis `,` $split_sizes attr-dict `:` functional-type(operands, results)
  }];
}

def Mrt_MulOp : Mrt_Op<"mul", [Pure]> { 
  let summary = "Multiplication operation";
  let description = [{
    Element-wise multiplication of two tensors.
  }];

  let arguments = (ins
    AnyRankedTensor:$lhs,
    AnyRankedTensor:$rhs
  );

  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` functional-type(operands, results)
  }];

  let generated = 1;
}

def Mrt_MoeInitRoutingV3Op : Mrt_Op<"moe_init_routing_v3"> {
  let summary = "aclnnMoeInitRoutingV3";
  let description = [{
    Operation for initializing Moe (Mixture of Experts) routing V3 using aclnn backend.
    Accepts input tensor, expert indices, scale/offset (optional), and configuration arguments.
    Produces expanded tensors for X, row indices, the expert token count or cumsum, and scale.
  }];

  let arguments = (ins
    AnyRankedTensor:$x,
    AnyRankedTensor:$expertIdx,
    AnyRankedTensor:$scaleOptional,
    AnyRankedTensor:$offsetOptional,
    Mrt_I64Type:$activeNum,
    Mrt_I64Type:$expertCapacity,
    Mrt_I64Type:$expertNum,
    Mrt_I64Type:$dropPadMode,
    Mrt_I64Type:$expertTokensNumType,
    Mrt_BooleanType:$expertTokensNumFlag,
    Mrt_I64Type:$quantMode,
    Mrt_I64ArrayType:$activeExpertRangeOptional,
    Mrt_I64Type:$rowIdxType
  );

  let results = (outs
    AnyRankedTensor:$expandedXOut,
    AnyRankedTensor:$expandedRowIdxOut,
    AnyRankedTensor:$expertTokensCountOrCumsumOut,
    AnyRankedTensor:$expandedScaleOut
  );
  
  let assemblyFormat = [{
    $x `,` $expertIdx `,` $scaleOptional `,` $offsetOptional `,`
    $activeNum `,` $expertCapacity `,` $expertNum `,` $dropPadMode `,`
    $expertTokensNumType `,` $expertTokensNumFlag `,` $quantMode `,`
    $activeExpertRangeOptional `,` $rowIdxType
    attr-dict `:` functional-type(operands, results)
  }];
}

def Mrt_AddRmsNormOp : Mrt_Op<"add_rms_norm"> {
  let summary = "aclnnAddRmsNorm";
  let description = [{
    Operation for fused Add and RmsNorm computation.
    Accepts two input tensors for addition, a gamma scaling factor tensor, and epsilon value for numerical stability.
    Produces normalized output tensor, the standard deviation, and the intermediate addition result.
  }];

  let arguments = (ins
    AnyRankedTensor:$x1,
    AnyRankedTensor:$x2,
    AnyRankedTensor:$gamma,
    Mrt_F32Type:$epsilon
  );

  let results = (outs
    AnyRankedTensor:$yOut,
    AnyRankedTensor:$rstdOut,
    AnyRankedTensor:$xOut
  );

  let assemblyFormat = [{
    $x1 `,` $x2 `,` $gamma `,` $epsilon
    attr-dict `:` functional-type(operands, results)
  }];

  let generated = 1;
}

def Mrt_RmsNormOp : Mrt_Op<"rms_norm"> {
  let summary = "aclnnRmsNorm";
  let description = [{
    Operation for RmsNorm computation.
    Accepts input tensor, gamma scaling factor, and epsilon value for numerical stability.
    Produces normalized output tensor and reciprocal of standard deviation.
  }];

  let arguments = (ins
    AnyRankedTensor:$x,
    AnyRankedTensor:$gamma,
    Mrt_F32Type:$epsilon
  );

  let results = (outs
    AnyRankedTensor:$yOut,
    AnyRankedTensor:$rstdOut
  );

  let assemblyFormat = [{
    $x `,` $gamma `,` $epsilon
    attr-dict `:` functional-type(operands, results)
  }];

  let generated = 1;
}

#endif // MRT_DIALECT_MRT_OPS_TD