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

class Mrt_Attr<string name, string attrMnemonic> : AttrDef<Mrt_Dialect, name> {
  let mnemonic = attrMnemonic;
}

class Mrt_Op<string mnemonic, list<Trait> traits = []> : 
    Op<Mrt_Dialect, mnemonic, traits> {
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

def Mrt_FloatType : Mrt_Type<"Float", "float"> {
  let summary = "Floating point type for MRT dialect";
  let description = [{
    Represents floating point numbers with specified precision.
  }];
  
  let parameters = (ins
    OptionalParameter<"unsigned">:$width
  );
  
  let assemblyFormat = [{
    `<` $width `>`
  }];
}

def Mrt_PrecisionAttr : Mrt_Attr<"Precision", "precision"> {
  let summary = "Precision attribute for MRT operations";
  let description = [{
    Specifies the numerical precision for computations.
  }];
  
  let parameters = (ins
    OptionalParameter<"std::string">:$value
  );
  
  let assemblyFormat = [{
    `<` $value `>`
  }];
}

def Mrt_ConvOp : Mrt_Op<"conv", [Pure]> {
  let summary = "2D convolution operation";
  let description = [{
    Performs a 2D convolution operation on input tensors.
  }];
  
  let arguments = (ins
    AnyRankedTensor:$input,
    AnyRankedTensor:$kernel,
    Optional<AnyRankedTensor>:$bias,
    I64ArrayAttr:$strides,
    I64ArrayAttr:$padding,
    I64ArrayAttr:$dilation,
    BoolAttr:$has_bias
  );
  
  let results = (outs AnyRankedTensor:$result);
  
  let assemblyFormat = [{
    $input `,` $kernel (`,` $bias^)? 
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
    AnyRankedTensor:$rhs,
    Optional<AnyRankedTensor>:$bias,
    BoolAttr:$transpose_a,
    BoolAttr:$transpose_b,
    BoolAttr:$has_bias
  );
  
  let results = (outs AnyRankedTensor:$result);
  
  let assemblyFormat = [{
    $lhs `,` $rhs (`,` $bias^)? 
    attr-dict `:` functional-type(operands, results)
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
    I64Attr:$axis
  );
  let results = (outs AnyRankedTensor:$result);
  
  let assemblyFormat = [{
    $input attr-dict `:` functional-type(operands, results)
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
    F32Attr:$epsilon,
    BoolAttr:$is_training
  );
  
  let results = (outs AnyRankedTensor:$result);
  
  let assemblyFormat = [{
    $input `,` $scale `,` $offset `,` $mean `,` $variance 
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
    I64ArrayAttr:$kernel_size,
    I64ArrayAttr:$strides,
    I64ArrayAttr:$padding,
    StrAttr:$pool_type
  );
  
  let results = (outs AnyRankedTensor:$result);
  
  let assemblyFormat = [{
    $input attr-dict `:` functional-type(operands, results)
  }];
}

def Mrt_ReshapeOp : Mrt_Op<"reshape", [Pure]> {
  let summary = "Reshape operation";
  let description = [{
    Reshapes input tensor to the specified shape.
  }];
  
  let arguments = (ins
    AnyRankedTensor:$input,
    I64ArrayAttr:$shape
  );
  
  let results = (outs AnyRankedTensor:$result);
  
  let assemblyFormat = [{
    $input `to` $shape attr-dict `:` functional-type(operands, results)
  }];
}

def Mrt_ConcatOp : Mrt_Op<"concat", [Pure]> {
  let summary = "Concatenation operation";
  let description = [{
    Concatenates tensors along the specified axis.
  }];
  
  let arguments = (ins
    Variadic<AnyRankedTensor>:$inputs,
    I64Attr:$axis
  );
  
  let results = (outs AnyRankedTensor:$result);
  
  let assemblyFormat = [{
    $inputs `along` `axis` $axis attr-dict `:` functional-type(operands, results)
  }];
}

def Mrt_SplitOp : Mrt_Op<"split", [Pure]> {
  let summary = "Split operation";
  let description = [{
    Splits tensor along the specified axis into multiple outputs.
  }];
  
  let arguments = (ins
    AnyRankedTensor:$input,
    I64Attr:$axis,
    I64ArrayAttr:$split_sizes
  );
  
  let results = (outs Variadic<AnyRankedTensor>:$results);
  
  let assemblyFormat = [{
    $input `along` `axis` $axis attr-dict `:` functional-type(operands, results)
  }];
}

#endif // MRT_DIALECT_MRT_OPS_TD