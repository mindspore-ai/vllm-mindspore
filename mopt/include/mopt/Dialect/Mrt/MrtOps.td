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

include "mopt/Dialect/Mrt/MrtTypes.td"

//===----------------------------------------------------------------------===//
// MRT Operations
//
// NOTE: All operations in this file are sorted alphabetically by their class name
// (the identifier after "def", e.g., Mrt_AddRmsNormOp, Mrt_BatchNormOp).
// When adding new operations, please maintain this alphabetical order to keep
// the file organized and easy to navigate.
//===----------------------------------------------------------------------===//

def Mrt_AddOp : Mrt_Op<"add", [Pure]> {
  let summary = "aclnnAdd";
  let description = [{
    Element-wise addition of two tensors with an optional alpha scaling factor.
    Computes: result = x + y * alpha
  }];

  let arguments = (ins
    MrtAnyTensor:$x,
    MrtAnyTensor:$y,
    Mrt_ScalarType:$alpha
  );

  let results = (outs MrtAnyTensor:$result);

  let assemblyFormat = [{
    $x `,` $y `,` $alpha attr-dict `:` functional-type(operands, results)
  }];

  let genAclnnOp = "aclnnAdd";
}

def Mrt_AddRmsNormOp : Mrt_Op<"add_rms_norm", [Pure]> {
  let summary = "aclnnAddRmsNorm";
  let description = [{
    Operation for fused Add and RmsNorm computation.
    Accepts two input tensors for addition, a gamma scaling factor tensor, and epsilon value for numerical stability.
    Produces normalized output tensor, the standard deviation, and the intermediate addition result.
  }];

  let arguments = (ins
    MrtAnyTensor:$x1,
    MrtAnyTensor:$x2,
    MrtAnyTensor:$gamma,
    Mrt_F64Type:$epsilon
  );

  let results = (outs
    MrtAnyTensor:$yOut,
    MrtAnyTensor:$rstdOut,
    MrtAnyTensor:$xOut
  );

  let assemblyFormat = [{
    $x1 `,` $x2 `,` $gamma `,` $epsilon
    attr-dict `:` functional-type(operands, results)
  }];

  let genAclnnOp = "aclnnAddRmsNorm";
}

def Mrt_ApplyRotaryPosEmbOp : Mrt_Op<"apply_rotary_pos_emb"> {
  let summary = "aclnnApplyRotaryPosEmb";
  let description = [{
    Operation for rope computation.
    Accepts query, key, cos, sin  tensor, and layout string for rope calculation.
    Produces the result after rope.
  }];

  let arguments = (ins
    MrtAnyTensor:$query,
    MrtAnyTensor:$key,
    MrtAnyTensor:$cos,
    MrtAnyTensor:$sin,
    Mrt_I64Type:$layout
  );

  let results = (outs
    MrtAnyTensor:$fake_query_out,
    MrtAnyTensor:$fake_key_out
  );

  let assemblyFormat = [{
    $query `,` $key `,` $cos `,` $sin `,` $layout
    attr-dict `:` functional-type(operands, results)
  }];

  let genAclnnOp = "aclnnApplyRotaryPosEmb";
  let genAclnnCfg = (AclnnConfig (IORef 0, 0), (IORef 1, 1));
}

def Mrt_BatchNormOp : Mrt_Op<"batch_norm", [Pure]> {
  let summary = "Batch normalization operation";
  let description = [{
    Performs batch normalization on input tensors.
  }];

  let arguments = (ins
    MrtAnyTensor:$input,
    MrtAnyTensor:$scale,
    MrtAnyTensor:$offset,
    MrtAnyTensor:$mean,
    MrtAnyTensor:$variance,
    Mrt_F64Type:$epsilon,
    Mrt_BooleanType:$is_training
  );

  let results = (outs MrtAnyTensor:$result);

  let assemblyFormat = [{
    $input `,` $scale `,` $offset `,` $mean `,` $variance `,` $epsilon `,` $is_training
    attr-dict `:` functional-type(operands, results)
  }];
}

def Mrt_BitwiseAndTensorOp : Mrt_Op<"bitwise_and_tensor", [Pure]> {
  let summary = "aclnnBitwiseAndTensor";
  let description = [{
    Operation for element-wise bitwise AND computation between two tensors.
    Performs bitwise AND operation on corresponding elements of input tensors.
    Both input tensors must have the same data type and compatible shapes for broadcasting.
  }];

  let arguments = (ins
    MrtAnyTensor:$self,
    MrtAnyTensor:$other
  );

  let results = (outs
    MrtAnyTensor:$out
  );

  let assemblyFormat = [{
    $self `,` $other
    attr-dict `:` functional-type(operands, results)
  }];

  let genAclnnOp = "aclnnBitwiseAndTensor";
}

def Mrt_BitwiseNotOp : Mrt_Op<"bitwise_not", [Pure]> {
  let summary = "aclnnBitwiseNot";
  let description = [{
    Operation for element-wise bitwise NOT computation on a tensor.
    Performs bitwise NOT operation on each element of the input tensor.
    The input tensor must have integer data type.
  }];

  let arguments = (ins
    MrtAnyTensor:$self
  );

  let results = (outs
    MrtAnyTensor:$out
  );

  let assemblyFormat = [{
    $self
    attr-dict `:` functional-type(operands, results)
  }];

  let genAclnnOp = "aclnnBitwiseNot";
}

def Mrt_CastOp : Mrt_Op<"cast", [Pure]> {
  let summary = "aclnnCast";
  let description = [{
    Casts a tensor to the specified data type.
  }];

  let arguments = (ins
    MrtAnyTensor:$input,
    Mrt_DtypeType:$dtype
  );

  let results = (outs MrtAnyTensor:$result);

  let assemblyFormat = [{
    $input `,` $dtype attr-dict `:` functional-type(operands, results)
  }];

  let genAclnnOp = "aclnnCast";
}

def Mrt_CatOp : Mrt_Op<"cat", [Pure]> {
  let summary = "Concatenation operation";
  let description = [{
    Concatenates tensors along the specified axis.
  }];

  let arguments = (ins
    MrtTensorList:$inputs,
    Mrt_I64Type:$axis
  );

  let results = (outs MrtAnyTensor:$result);

  let assemblyFormat = [{
    $inputs `,` $axis attr-dict `:` functional-type(operands, results)
  }];
  let genAclnnOp = "aclnnCat";
}

def Mrt_CloneOp : Mrt_Op<"clone", [Pure]> {
  let summary = "Clone operation";
  let description = [{
    Returns a clone of the input tensor.
  }];

  let arguments = (ins
    MrtAnyTensor:$self
  );

  let results = (outs MrtAnyTensor:$result);

  let assemblyFormat = [{
    $self attr-dict `:` functional-type(operands, results)
  }];
}

def Mrt_ConvOp : Mrt_Op<"conv", [Pure]> {
  let summary = "2D convolution operation";
  let description = [{
    Performs a 2D convolution operation on input tensors.
  }];

  let arguments = (ins
    MrtAnyTensor:$input,
    MrtAnyTensor:$kernel,
    MrtOptional<MrtAnyTensor>:$bias,
    Mrt_I64ArrayType:$strides,
    Mrt_I64ArrayType:$padding,
    Mrt_I64ArrayType:$dilation,
    Mrt_BooleanType:$has_bias
  );

  let results = (outs MrtAnyTensor:$result);

  let assemblyFormat = [{
    $input `,` $kernel `,` $bias `,` $strides `,` $padding `,` $dilation `,` $has_bias
    attr-dict `:` functional-type(operands, results)
  }];
}

def Mrt_DivOp : Mrt_Op<"div", [Pure]> {
  let summary = "aclnnDiv";
  let description = [{
    Operation for element-wise division of two tensors.
    Performs division of corresponding elements of input tensors.
    Both input tensors must have compatible shapes for broadcasting.
  }];

  let arguments = (ins
    MrtAnyTensor:$self,
    MrtAnyTensor:$other
  );

  let results = (outs
    MrtAnyTensor:$out
  );

  let assemblyFormat = [{
    $self `,` $other
    attr-dict `:` functional-type(operands, results)
  }];

  let genAclnnOp = "aclnnDiv";
}

def Mrt_DivModOp : Mrt_Op<"div_mod", [Pure]> {
  let summary = "aclnnDivMod";
  let description = [{
    Operation for element-wise division and modulus computation between two tensors.
    Performs division and modulus operations on corresponding elements of input tensors.
    The mode parameter controls the rounding behavior for integer division.
    Both input tensors must have compatible shapes for broadcasting.
  }];

  let arguments = (ins
    MrtAnyTensor:$self,
    MrtAnyTensor:$other,
    Mrt_I64Type:$mode
  );

  let results = (outs
    MrtAnyTensor:$out
  );

  let assemblyFormat = [{
    $self `,` $other `,` $mode
    attr-dict `:` functional-type(operands, results)
  }];

  let genAclnnOp = "aclnnDivMod";
}

def Mrt_EmbeddingOp : Mrt_Op<"embedding", [Pure]> {
  let summary = "aclnnEmbedding";
  let description = [{
    Operation for embedding computation.
    Accepts weight, indices tensor.
    Produces embedding output.
  }];

  let arguments = (ins
    MrtAnyTensor:$weight,
    MrtAnyTensor:$indices
  );

  let results = (outs
    MrtAnyTensor:$out
  );

  let genAclnnOp = "aclnnEmbedding";
}

def Mrt_EmptyOp : Mrt_Op<"empty", [
    DeclareOpInterfaceMethods<InferTypeOpInterface>
  ]> {
  let summary = "";
  let description = [{
    Operation for empty ops.
    Accept shape size.
    Produces the device the same shape tensor.
  }];

  let arguments = (ins
    Mrt_I64ArrayType:$size,
    Mrt_DtypeType:$dtype,
    Mrt_StringType:$device
  );

  let results = (outs
    MrtAnyTensor:$yOut
  );

  let assemblyFormat = [{
    $size `,` $dtype `,` $device
    attr-dict `:` functional-type(operands, results)
  }];
}

def Mrt_EqOp : Mrt_Op<"eq", [Pure]> {
  let summary = "Element-wise equality comparison";
  let description = [{
    Element-wise equality comparison of two tensors.
    Returns a boolean tensor where each element is true if the corresponding elements are equal.
  }];

  let arguments = (ins
    MrtAnyTensor:$self,
    MrtAnyTensor:$other
  );

  let results = (outs MrtAnyTensor:$out);

  let assemblyFormat = [{
    $self `,` $other attr-dict `:` functional-type(operands, results)
  }];

  let genAclnnOp = "aclnnEqTensor";
}

def Mrt_ExpandOp : Mrt_Op<"expand", [Pure]> {
  let summary = "expand operation";
  let description = [{
    Expands the input tensor 'self' to the given 'size' by broadcasting.
    The output tensor will have the specified shape.
  }];

  let arguments = (ins
    MrtAnyTensor:$self,
    Mrt_I64ArrayType:$size
  );

  let results = (outs MrtAnyTensor:$out);

  let assemblyFormat = [{
    $self `,` $size attr-dict `:` functional-type(operands, results)
  }];

  let genAclnnOp = "aclnnExpand";
}

def Mrt_FlattenOp : Mrt_Op<"flatten", [Pure]> {
  let summary = "flatten operation";
  let description = [{
    Flatten input tensor to the specified shape.
    Takes two operands: input tensor and shape.
  }];

  let arguments = (ins
    MrtAnyTensor:$input,
    Mrt_I64Type:$start_idx,
    Mrt_I64Type:$end_idx
  );

  let results = (outs MrtAnyTensor:$result);

  let assemblyFormat = [{
    $input `,` $start_idx `,` $end_idx attr-dict `:` functional-type(operands, results)
  }];
}

def Mrt_GeOp : Mrt_Op<"ge", [Pure]> {
  let summary = "Element-wise greater-than-or-equal comparison";
  let description = [{
    Element-wise greater-than-or-equal comparison of two tensors.
    Returns a boolean tensor where each element is true if the corresponding element in self is greater than or equal to the element in other.
  }];

  let arguments = (ins
    MrtAnyTensor:$self,
    MrtAnyTensor:$other
  );

  let results = (outs MrtAnyTensor:$out);

  let assemblyFormat = [{
    $self `,` $other attr-dict `:` functional-type(operands, results)
  }];

  let genAclnnOp = "aclnnGeTensor";
}

def Mrt_GroupedMatmulOp : Mrt_Op<"grouped_matmul", [Pure]> {
  let summary = "NPU grouped matmul operation";
  let description = [{
    Performs grouped matrix multiplication on NPU with optional bias, scale, offset, and activation.
  }];

  let arguments = (ins
    MrtTensorList:$x,
    MrtTensorList:$weight,
    MrtOptional<MrtTensorList>:$bias,
    MrtOptional<MrtTensorList>:$scale,
    MrtOptional<MrtTensorList>:$offset,
    MrtOptional<MrtTensorList>:$antiquant_scale,
    MrtOptional<MrtTensorList>:$antiquant_offset,
    MrtOptional<MrtTensorList>:$per_token_scale,
    MrtOptional<MrtAnyTensor>:$group_list,
    MrtOptional<MrtTensorList>:$activation_input,
    MrtOptional<MrtTensorList>:$activation_quant_scale,
    MrtOptional<MrtTensorList>:$activation_quant_offset,
    Mrt_I64Type:$split_item,
    Mrt_I64Type:$group_type,
    Mrt_I64Type:$group_list_type,
    Mrt_I64Type:$act_type,
    MrtOptional<Mrt_I64ArrayType>:$tuning_config,
    Mrt_DtypeType:$output_dtype
  );

  let results = (outs
    MrtTensorList:$output
  );

  let assemblyFormat = [{
    $x `,` $weight `,` $bias `,` $scale `,` $offset `,` $antiquant_scale `,` $antiquant_offset `,` $per_token_scale 
    `,` $group_list `,` $activation_input `,` $activation_quant_scale `,` $activation_quant_offset
    `,` $split_item `,` $group_type `,` $group_list_type `,` $act_type `,` $tuning_config `,` $output_dtype
    attr-dict `:` functional-type(operands, results)
  }];
}

def Mrt_GtOp : Mrt_Op<"gt", [Pure]> {
  let summary = "Element-wise greater-than comparison";
  let description = [{
    Element-wise greater-than comparison of two tensors.
    Returns a boolean tensor where each element is true if the corresponding element in self is greater than the element in other.
  }];

  let arguments = (ins
    MrtAnyTensor:$self,
    MrtAnyTensor:$other
  );

  let results = (outs MrtAnyTensor:$out);

  let assemblyFormat = [{
    $self `,` $other attr-dict `:` functional-type(operands, results)
  }];

  let genAclnnOp = "aclnnGtTensor";
}

def Mrt_IndexTensorOp : Mrt_Op<"index_tensor", [Pure]> {
  let summary = "Indexes a tensor using a list of index tensors.";
  let description = [{
    Operation for indexing a tensor using a list of index tensors.
    Performs advanced indexing where each tensor in the indices list
    corresponds to indexing along that dimension.
    This is the implementation of `aten.index.Tensor_hacked_twin`.
  }];

  let arguments = (ins
    MrtAnyTensor:$self,
    MrtTensorList:$indices
  );

  let results = (outs MrtAnyTensor:$result);

  let assemblyFormat = [{
    $self `,` $indices attr-dict `:` functional-type(operands, results)
  }];

  let genAclnnOp = "aclnnIndex";
}

def Mrt_InplaceCopyOp : Mrt_Op<"inplace_copy"> {
  let summary = "Copy operation";
  let description = [{
    Copies data from source tensor to destination tensor. The optional boolean parameter controls whether the copy is in-place.
  }];

  let arguments = (ins
    MrtAnyTensor:$dst,
    MrtAnyTensor:$src,
    Mrt_BooleanType:$non_blocking
  );

  let results = (outs MrtAnyTensor:$result);

  let assemblyFormat = [{
    $dst `,` $src `,` $non_blocking
    attr-dict `:` functional-type(operands, results)
  }];
}

def Mrt_InplaceFillScalarOp : Mrt_Op<"inplace_fill_scalar"> {
  let summary = "Fill tensor with scalar value";
  let description = [{
    Fills the destination tensor with the specified scalar value.
    Computes: result = fill(x, value)
  }];

  let arguments = (ins
    MrtAnyTensor:$x,
    Mrt_ScalarType:$value
  );

  let results = (outs MrtAnyTensor:$result);

  let assemblyFormat = [{
    $x `,` $value attr-dict `:` functional-type(operands, results)
  }];

  let genAclnnOp = "aclnnInplaceFillScalar";
  let genAclnnCfg = (AclnnConfig (IORef 0, 0));
}

def Mrt_InplaceFillTensorOp : Mrt_Op<"inplace_fill_tensor"> {
  let summary = "Fill tensor with tensor value";
  let description = [{
    Fills the destination tensor with the specified tensor value.
    Computes: result = fill(x, value)
  }];

  let arguments = (ins
    MrtAnyTensor:$x,
    MrtAnyTensor:$value
  );

  let results = (outs MrtAnyTensor:$result);

  let assemblyFormat = [{
    $x `,` $value attr-dict `:` functional-type(operands, results)
  }];

  let genAclnnOp = "aclnnInplaceFillTensor";
  let genAclnnCfg = (AclnnConfig (IORef 0, 0));
}

def Mrt_InplaceMaskedFillScalarOp : Mrt_Op<"inplace_masked_fill_scalar"> {
  let summary = "Masked fill operation";
  let description = [{
    Replaces elements in the input tensor with the fill value where the mask is True.
  }];

  let arguments = (ins
    MrtAnyTensor:$input,
    MrtAnyTensor:$mask,
    Mrt_ScalarType:$value
  );

  let results = (outs MrtAnyTensor:$result);

  let assemblyFormat = [{
    $input `,` $mask `,` $value attr-dict `:` functional-type(operands, results)
  }];

  let genAclnnOp = "aclnnInplaceMaskedFillScalar";
  let genAclnnCfg = (AclnnConfig (IORef 0, 0));
}

def Mrt_InplaceMaskedFillTensorOp : Mrt_Op<"inplace_masked_fill_tensor"> {
  let summary = "Masked fill operation";
  let description = [{
    Replaces elements in the input tensor with the fill value where the mask is True.
  }];

  let arguments = (ins
    MrtAnyTensor:$input,
    MrtAnyTensor:$mask,
    MrtAnyTensor:$value
  );

  let results = (outs MrtAnyTensor:$result);

  let assemblyFormat = [{
    $input `,` $mask `,` $value attr-dict `:` functional-type(operands, results)
  }];

  let genAclnnOp = "aclnnInplaceMaskedFillTensor";
  let genAclnnCfg = (AclnnConfig (IORef 0, 0));
}

def Mrt_LeOp : Mrt_Op<"le", [Pure]> {
  let summary = "Element-wise less-than-or-equal comparison";
  let description = [{
    Element-wise less-than-or-equal comparison of two tensors.
    Returns a boolean tensor where each element is true if the corresponding element in self is less than or equal to the element in other.
  }];

  let arguments = (ins
    MrtAnyTensor:$self,
    MrtAnyTensor:$other
  );

  let results = (outs MrtAnyTensor:$out);

  let assemblyFormat = [{
    $self `,` $other attr-dict `:` functional-type(operands, results)
  }];

  let genAclnnOp = "aclnnLeTensor";
}

def Mrt_LinalgCallOp : Mrt_Op<"linalg_call"> {
  let summary = "Invoke a Linalg MLIR kernel";
  let description = [{
    Invokes a Linalg MLIR kernel compiled through BiSheng compiler.
    The mlir_text operand contains a serialized Linalg MLIR module with a
    function marked with hacc.entry attribute. 

    Internal workflow:
    1. Init: Extract MLIR text, call LoweredOpHelper::CreateFromMlirText()
    2. InferShape/CalcWorkspace/Launch: Delegate to internal AutoLoweredOp
  }];

  let arguments = (ins
    Mrt_StringType:$mlir_text,
    Variadic<MrtAnyTensor>:$inputs
  );

  let results = (outs
    Variadic<MrtAnyTensor>:$outputs
  );

  let assemblyFormat = [{
    $mlir_text `,` $inputs attr-dict `:` functional-type(operands, results)
  }];
}

def Mrt_LinearOp : Mrt_Op<"linear", [Pure]> {
  let summary = "dense";
  let description = [{
    Operation for linear computation.
    Accepts weight, indices tensor.
    Produces embedding output.
  }];

  let arguments = (ins
    MrtAnyTensor:$x,
    MrtAnyTensor:$weight,
    Optional<MrtAnyTensor>:$bias
  );

  let results = (outs
    MrtAnyTensor:$out
  );

  let assemblyFormat = [{
    $x `,` $weight (`,` $bias^)?
    attr-dict `:` functional-type(operands, results)
  }];
}

def Mrt_LtOp : Mrt_Op<"lt", [Pure]> {
  let summary = "Element-wise less-than comparison";
  let description = [{
    Element-wise less-than comparison of two tensors.
    Returns a boolean tensor where each element is true if the corresponding element in self is less than the element in other.
  }];

  let arguments = (ins
    MrtAnyTensor:$self,
    MrtAnyTensor:$other
  );

  let results = (outs MrtAnyTensor:$out);

  let assemblyFormat = [{
    $self `,` $other attr-dict `:` functional-type(operands, results)
  }];

  let genAclnnOp = "aclnnLtTensor";
}

def Mrt_MakeListOp : Mrt_Op<"make_list", [Pure]> {
  let summary = "Construct a list from elements";
  let description = [{
    Creates a list from a list of elements, similar to torch.prim.ListConstruct.
    This operation takes variadic elements and constructs a list type containing them.
  }];

  let arguments = (ins
    Variadic<AnyType>:$elements
  );

  let results = (outs MrtAnyList:$result);

  let assemblyFormat = [{
    $elements attr-dict `:` functional-type(operands, results)
  }];
}

def Mrt_MaskedFillScalarOp : Mrt_Op<"masked_fill_scalar", [Pure]> {
  let summary = "Masked fill operation";
  let description = [{
    Replaces elements in the input tensor with the fill value where the mask is True.
  }];

  let arguments = (ins
    MrtAnyTensor:$input,
    MrtAnyTensor:$mask,
    Mrt_ScalarType:$value
  );

  let results = (outs MrtAnyTensor:$result);

  let assemblyFormat = [{
    $input `,` $mask `,` $value attr-dict `:` functional-type(operands, results)
  }];
}

def Mrt_MaskedFillTensorOp : Mrt_Op<"masked_fill_tensor", [Pure]> {
  let summary = "Masked fill operation";
  let description = [{
    Replaces elements in the input tensor with the fill value where the mask is True.
  }];

  let arguments = (ins
    MrtAnyTensor:$input,
    MrtAnyTensor:$mask,
    MrtAnyTensor:$value
  );

  let results = (outs MrtAnyTensor:$result);

  let assemblyFormat = [{
    $input `,` $mask `,` $value attr-dict `:` functional-type(operands, results)
  }];
}

def Mrt_Matmul : Mrt_Op<"matmul", [Pure]> {
  let summary = "aclnnMatmul";
  let description = [{
    Performs matrix multiplication of two tensors (supports 1D to 6D tensor).
  }];

  let arguments = (ins
    MrtAnyTensor:$self,
    MrtAnyTensor:$other
  );

  let results = (outs MrtAnyTensor:$result);

  let assemblyFormat = [{
    $self `,` $other attr-dict `:` functional-type(operands, results)
  }];
}

def Mrt_Mm : Mrt_Op<"mm", [Pure]> {
  let summary = "aclnnMm";
  let description = [{
    Performs matrix multiplication of two 2D tensors.
  }];

  let arguments = (ins
    MrtAnyTensor:$self,
    MrtAnyTensor:$other
  );

  let results = (outs MrtAnyTensor:$result);

  let assemblyFormat = [{
    $self `,` $other attr-dict `:` functional-type(operands, results)
  }];
}

def Mrt_MoeGatingTopKSoftmaxOp : Mrt_Op<"moe_gating_top_k_softmax", [Pure]> {
  let summary = "moe calc for input do Softmax, and do topK selection";
  let description = [{
    Performs a Softmax and TopK on input tensors.
  }];
  
  let arguments = (ins
    MrtAnyTensor:$x,
    MrtOptional<MrtAnyTensor>:$finished,
    Mrt_I64Type:$k
  );
  
  let results = (outs 
    MrtAnyTensor:$y,
    MrtAnyTensor:$expert_idx,
    MrtAnyTensor:$row_idx
  );
  
  let assemblyFormat = [{
    $x `,` $finished `,` $k
    attr-dict `:` functional-type(operands, results)
  }];

  let genAclnnOp = "aclnnMoeGatingTopKSoftmax";
}

def Mrt_MoeInitRoutingV3Op : Mrt_Op<"moe_init_routing_v3", [Pure]> {
  let summary = "aclnnMoeInitRoutingV3";
  let description = [{
    Operation for initializing Moe (Mixture of Experts) routing V3 using aclnn backend.
    Accepts input tensor, expert indices, scale/offset (optional), and configuration arguments.
    Produces expanded tensors for X, row indices, the expert token count or cumsum, and scale.
  }];

  let arguments = (ins
    MrtAnyTensor:$x,
    MrtAnyTensor:$expertIdx,
    MrtOptional<MrtAnyTensor>:$scaleOptional,
    MrtOptional<MrtAnyTensor>:$offsetOptional,
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
    MrtAnyTensor:$expandedXOut,
    MrtAnyTensor:$expandedRowIdxOut,
    MrtAnyTensor:$expertTokensCountOrCumsumOut,
    MrtAnyTensor:$expandedScaleOut
  );

  let assemblyFormat = [{
    $x `,` $expertIdx `,` $scaleOptional `,` $offsetOptional `,`
    $activeNum `,` $expertCapacity `,` $expertNum `,` $dropPadMode `,`
    $expertTokensNumType `,` $expertTokensNumFlag `,` $quantMode `,`
    $activeExpertRangeOptional `,` $rowIdxType
    attr-dict `:` functional-type(operands, results)
  }];

  let genAclnnOp = "aclnnMoeInitRoutingV3";
}

def Mrt_MoeTokenUnpermuteOp : Mrt_Op<"moe_token_unpermute", [Pure]> {
  let summary = "aclnnMoeTokenUnpermute";
  let description = [{
    Operation for moe token unpermute computation.
    Accepts permuted_tokens, sorted_indices and optional probs tensor, padded_mode, optional restore_shape.
    Produces unpermute_tokens as result.
  }];

  let arguments = (ins
    MrtAnyTensor:$permuted_tokens,
    MrtAnyTensor:$sorted_indices,
    MrtOptional<MrtAnyTensor>:$probs,
    Mrt_BooleanType:$padded_mode,
    MrtOptional<Mrt_I64ArrayType>:$restore_shape
  );

  let results = (outs
    MrtAnyTensor:$Out
  );

  let assemblyFormat = [{
    $permuted_tokens `,` $sorted_indices `,` $probs `,` $padded_mode `,` $restore_shape
    attr-dict `:` functional-type(operands, results)
  }];

  let genAclnnOp = "aclnnMoeTokenUnpermute";
}

def Mrt_MulOp : Mrt_Op<"mul", [Pure, Commutative]> {
  let summary = "Multiplication operation";
  let description = [{
    Element-wise multiplication of two tensors.
  }];

  let arguments = (ins
    MrtAnyTensor:$lhs,
    MrtAnyTensor:$rhs
  );

  let results = (outs MrtAnyTensor:$result);

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` functional-type(operands, results)
  }];

  let genAclnnOp = "aclnnMul";
}

def Mrt_NeOp : Mrt_Op<"ne", [Pure]> {
  let summary = "Element-wise not-equal comparison";
  let description = [{
    Element-wise not-equal comparison of two tensors.
    Returns a boolean tensor where each element is true if the corresponding elements are not equal.
  }];

  let arguments = (ins
    MrtAnyTensor:$self,
    MrtAnyTensor:$other
  );

  let results = (outs MrtAnyTensor:$out);

  let assemblyFormat = [{
    $self `,` $other attr-dict `:` functional-type(operands, results)
  }];

  let genAclnnOp = "aclnnNeTensor";
}

def Mrt_PermuteOp : Mrt_Op<"permute", [Pure]> {
  let summary = "Tensor permutation operation";
  let description = [{
    Permutes the dimensions of a tensor according to the specified order.
    Reorders the axes of the input tensor based on the permutation array.
  }];

  let arguments = (ins
    MrtAnyTensor:$input,
    Mrt_I64ArrayType:$perm
  );

  let results = (outs MrtAnyTensor:$output);

  let assemblyFormat = [{
    $input `,` $perm attr-dict `:` functional-type(operands, results)
  }];

  let genAclnnOp = "aclnnPermute";
}

def Mrt_PoolOp : Mrt_Op<"pool", [Pure]> {
  let summary = "Pooling operation";
  let description = [{
    Performs pooling operations (max, average, etc.) on input tensors.
  }];

  let arguments = (ins
    MrtAnyTensor:$input,
    Mrt_I64ArrayType:$kernel_size,
    Mrt_I64ArrayType:$strides,
    Mrt_I64ArrayType:$padding,
    Mrt_StringType:$pool_type
  );

  let results = (outs MrtAnyTensor:$result);

  let assemblyFormat = [{
    $input `,` $kernel_size `,` $strides `,` $padding `,` $pool_type
    attr-dict `:` functional-type(operands, results)
  }];
}

def Mrt_ReduceSumOp : Mrt_Op<"reduce_sum", [Pure]> {
  let summary = "Sum operation";
  let description = [{
    Computes the sum of tensor elements along specified dimensions.
    The boolean parameter controls whether to keep dimensions,
    and the optional dtype parameter specifies the output data type.
  }];

  let arguments = (ins
    MrtAnyTensor:$input,
    Mrt_I64ArrayType:$dimensions,
    Mrt_BooleanType:$keepdim,
    Mrt_DtypeType:$dtype
  );

  let results = (outs MrtAnyTensor:$result);

  let assemblyFormat = [{
    $input `,` $dimensions `,` $keepdim `,` $dtype
    attr-dict `:` functional-type(operands, results)
  }];

  let genAclnnOp = "aclnnReduceSum";
}

def Mrt_ReluOp : Mrt_Op<"relu", [Pure]> {
  let summary = "ReLU activation function";
  let description = [{
    Applies the ReLU (Rectified Linear Unit) activation function.
  }];

  let arguments = (ins MrtAnyTensor:$input);
  let results = (outs MrtAnyTensor:$result);

  let assemblyFormat = [{
    $input attr-dict `:` functional-type(operands, results)
  }];
}

def Mrt_RemainderTensorTensorOp : Mrt_Op<"remainder_tensor_tensor", [Pure]> {
  let summary = "aclnnRemainderTensorTensor";
  let description = [{
    Operation for element-wise remainder computation between two tensors.
    Performs remainder operation on corresponding elements of input tensors.
    Computes the remainder of division of elements in self tensor by elements in other tensor.
    Both input tensors must have compatible shapes for broadcasting.
  }];

  let arguments = (ins
    MrtAnyTensor:$self,
    MrtAnyTensor:$other
  );

  let results = (outs
    MrtAnyTensor:$out
  );

  let assemblyFormat = [{
    $self `,` $other
    attr-dict `:` functional-type(operands, results)
  }];

  let genAclnnOp = "aclnnRemainderTensorTensor";
}

def Mrt_ReshapeOp : Mrt_Op<"reshape", [Pure]> {
  let summary = "Reshape operation";
  let description = [{
    Reshapes input tensor to the specified shape.
    Takes two operands: input tensor and shape.
  }];

  let arguments = (ins
    MrtAnyTensor:$input,
    Mrt_I64ArrayType:$shape
  );

  let results = (outs MrtAnyTensor:$result);

  let assemblyFormat = [{
    $input `,` $shape attr-dict `:` functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

def Mrt_RmsNormOp : Mrt_Op<"rms_norm", [Pure]> {
  let summary = "aclnnRmsNorm";
  let description = [{
    Operation for RmsNorm computation.
    Accepts input tensor, gamma scaling factor, and epsilon value for numerical stability.
    Produces normalized output tensor and reciprocal of standard deviation.
  }];

  let arguments = (ins
    MrtAnyTensor:$x,
    MrtAnyTensor:$gamma,
    Mrt_F64Type:$epsilon
  );

  let results = (outs
    MrtAnyTensor:$yOut,
    MrtAnyTensor:$rstdOut
  );

  let assemblyFormat = [{
    $x `,` $gamma `,` $epsilon
    attr-dict `:` functional-type(operands, results)
  }];

  let genAclnnOp = "aclnnRmsNorm";
}

def Mrt_ScatterNdUpdateOp : Mrt_Op<"scatter_nd_update"> {
  let summary = "aclnnScatterNdUpdate";
  let description = [{
    Operation for ScatterNDUpdate computation.
    Accepts input, indices, updates tensors.
    Return the updated input tensor as result, which is a in-place ops.
  }];

  let arguments = (ins
    MrtAnyTensor:$input,
    MrtAnyTensor:$indices,
    MrtAnyTensor:$updates
  );

  let results = (outs
    MrtAnyTensor:$outs
  );

  let genAclnnOp = "aclnnScatterNdUpdate";
  let genAclnnCfg = (AclnnConfig (IORef 0, 0));
}

def Mrt_SigmoidOp : Mrt_Op<"sigmoid", [Pure]> {
  let summary = "Sigmoid activation function";
  let description = [{
    Applies the sigmoid activation function.
  }];

  let arguments = (ins MrtAnyTensor:$input);
  let results = (outs MrtAnyTensor:$result);

  let assemblyFormat = [{
    $input attr-dict `:` functional-type(operands, results)
  }];

  let genAclnnOp = "aclnnSigmoid";
}

def Mrt_SizeOp : Mrt_Op<"size", [Pure]> {
  let summary = "Get the size of a tensor dimension";
  let description = [{
    Returns the size of the specified dimension of the input tensor.
    Takes two operands: input tensor and dimension index.
    Returns a scalar tensor of type I64.
  }];

  let arguments = (ins
    MrtAnyTensor:$input,
    Mrt_I64Type:$dim
  );

  let results = (outs Mrt_I64Type:$result);

  let assemblyFormat = [{
    $input `,` $dim attr-dict `:` functional-type(operands, results)
  }];
}

def Mrt_SliceOp : Mrt_Op<"slice", [Pure]> {
  let summary = "Selects a range of elements from an input tensor along a given dimension.";
  let description = [{
    Extracts a contiguous slice from the input tensor `self` along the specified dimension `dim`, 
    beginning at index `start`, and continuing up to (but not including) index `end`, 
    with elements selected every `step` indices.
  }];

  let arguments = (ins
    MrtAnyTensor:$self,
    Mrt_I64Type:$dim,
    Mrt_I64Type:$start,
    Mrt_I64Type:$end,
    Mrt_I64Type:$step
  );

  let results = (outs
    MrtAnyTensor:$result
  );

  let assemblyFormat = [{
    $self `,` $dim `,` $start `,` $end `,` $step
    attr-dict `:` functional-type(operands, results)
  }];

  let genAclnnOp = "aclnnSlice";
}

def Mrt_SoftmaxOp : Mrt_Op<"softmax", [Pure]> {
  let summary = "Softmax activation function";
  let description = [{
    Applies the softmax activation function along the specified axis.
  }];

  let arguments = (ins
    MrtAnyTensor:$input,
    Mrt_I64Type:$axis
  );
  let results = (outs MrtAnyTensor:$result);

  let assemblyFormat = [{
    $input `,` $axis attr-dict `:` functional-type(operands, results)
  }];
}

def Mrt_SplitOp : Mrt_Op<"split_with_size", [Pure]> {
  let summary = "Split operation";
  let description = [{
    Splits tensor along the specified axis into multiple outputs.
  }];

  let arguments = (ins
    MrtAnyTensor:$input,
    Mrt_I64ArrayType:$split_sizes,
    Mrt_I64Type:$axis
  );

  let results = (outs MrtTensorList:$results);

  let assemblyFormat = [{
    $input `,` $split_sizes `,` $axis attr-dict `:` functional-type(operands, results)
  }];
  
  let genAclnnOp = "aclnnSplitWithSize";
}

def Mrt_SplitTensorOp : Mrt_Op<"split_tensor", [Pure]> {
  let summary = "Split tensor operation";
  let description = [{
    Splits tensor along the specified axis into multiple outputs.
  }];

  let arguments = (ins
    MrtAnyTensor:$input,
    Mrt_I64Type:$split_sizes,
    Mrt_I64Type:$axis
  );

  let results = (outs MrtTensorList:$results);

  let assemblyFormat = [{
    $input `,` $split_sizes `,` $axis attr-dict `:` functional-type(operands, results)
  }];

  let genAclnnOp = "aclnnSplitTensor";
}

def Mrt_SqueezeDimOp : Mrt_Op<"squeeze_dim", [Pure]> {
  let summary = "Removes a single dimension of size one from the input tensor at the specified position (dim).";
  let description = [{
    The returned tensor shares the same underlying data with this tensor.
  }];

  let arguments = (ins
    MrtAnyTensor:$input,
    Mrt_I64Type:$dim
  );

  let results = (outs
    MrtAnyTensor:$yOut
  );

  let assemblyFormat = [{
    $input `,` $dim attr-dict `:` functional-type(operands, results)
  }];
}

def Mrt_StridedSliceAssignOp : Mrt_Op<"strided_slice_assign"> {
  let summary = "Strided slice assign operation";
  let description = [{
    Strided slice operation with support for updating values in the input tensor.
    Extracts a slice from the input tensor along the specified axes with the given starts, ends, and strides,
    and updates the extracted slice with the update tensor.
  }];

  let arguments = (ins
    MrtAnyTensor:$x,
    MrtAnyTensor:$update,
    Mrt_I64ArrayType:$begin,
    Mrt_I64ArrayType:$end,
    Mrt_I64ArrayType:$strides,
    MrtOptional<Mrt_I64ArrayType>:$axes
  );

  let results = (outs);

  let assemblyFormat = [{
    $x `,` $update `,` $begin `,` $end `,` $strides `,` $axes attr-dict `:` type(operands)
  }];
}

def Mrt_SubOp : Mrt_Op<"sub", [Pure]> {
  let summary = "aclnnSub";
  let description = [{
    Element-wise subtraction of two tensors with an optional alpha scaling factor.
    Computes: result = x - y * alpha
  }];

  let arguments = (ins
    MrtAnyTensor:$x,
    MrtAnyTensor:$y,
    Mrt_ScalarType:$alpha
  );

  let results = (outs MrtAnyTensor:$result);

  let assemblyFormat = [{
    $x `,` $y `,` $alpha attr-dict `:` functional-type(operands, results)
  }];

  let genAclnnOp = "aclnnSub";
}

def Mrt_SwiGluOp : Mrt_Op<"swiglu"> {
  let summary = "aclnnSwiGlu";
  let description = [{
    Operation for swiglu computation.
    Accepts x tensor and the split dim.
    Produces swiglu output.
  }];

  let arguments = (ins
    MrtAnyTensor:$x,
    Mrt_I64Type:$dim
  );

  let results = (outs
    MrtAnyTensor:$out
  );
  let assemblyFormat = [{
    $x `,` $dim
    attr-dict `:` functional-type(operands, results)
  }];

  let genAclnnOp = "aclnnSwiGlu";
}

def Mrt_UnpackListOp : Mrt_Op<"unpack_list", [Pure]> {
  let summary = "Unpack a list into individual elements";
  let description = [{
    Unpacks a list into individual elements, similar to torch.prim.ListUnpack.
    This operation takes a list as input and produces variadic outputs representing
    each element of the list. The number of outputs must match the list length.
  }];

  let arguments = (ins
    MrtAnyList:$list
  );

  let results = (outs
    Variadic<AnyType>:$elements
  );

  let assemblyFormat = [{
    $list attr-dict `:` functional-type(operands, results)
  }];
}

def Mrt_UnsqueezeOp : Mrt_Op<"unsqueeze", [Pure]> {
  let summary = "Returns a new tensor with a dimension of size one inserted at the specified position.";
  let description = [{
    The returned tensor shares the same underlying data with this tensor.
  }];

  let arguments = (ins
    MrtAnyTensor:$input,
    Mrt_I64Type:$dim
  );

  let results = (outs MrtAnyTensor:$result);

  let assemblyFormat = [{
    $input `,` $dim attr-dict `:` functional-type(operands, results)
  }];
}

def Mrt_ViewOp : Mrt_Op<"view", [Pure]> {
  let summary = "Returns a view tensor which shares the same memory with the original one.";

  let arguments = (ins
    MrtAnyTensor:$input,
    Mrt_I64ArrayType:$shape
  );

  let results = (outs
    MrtAnyTensor:$yOut
  );

  let assemblyFormat = [{
    $input `,` $shape attr-dict `:` functional-type(operands, results)
  }];
}

def Mrt_ZerosOp : Mrt_Op<"zeros", [
    DeclareOpInterfaceMethods<InferTypeOpInterface>
  ]> {
  let summary = "aclnnInplaceZero";
  let description = [{
    Operation for zeros ops.
    Accept shape size.
    Produces the device the same shape tensor, the value in tensor has been set zero.
  }];

  let arguments = (ins
    Mrt_I64ArrayType:$size,
    Mrt_DtypeType:$dtype,
    Mrt_StringType:$device
  );

  let results = (outs
    MrtAnyTensor:$yOut
  );

  let assemblyFormat = [{
    $size `,` $dtype `,` $device
    attr-dict `:` functional-type(operands, results)
  }];
}

#endif // MRT_DIALECT_MRT_OPS_TD
