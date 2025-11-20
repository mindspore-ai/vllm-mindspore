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
    Mrt_F32Type:$epsilon
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

  let generated = 1;
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
    Mrt_F32Type:$epsilon,
    Mrt_BooleanType:$is_training
  );

  let results = (outs MrtAnyTensor:$result);

  let assemblyFormat = [{
    $input `,` $scale `,` $offset `,` $mean `,` $variance `,` $epsilon `,` $is_training
    attr-dict `:` functional-type(operands, results)
  }];
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

  let generated = 1;
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

  let results = (outs MrtAnyTensor:$result);

  let assemblyFormat = [{
    $inputs `,` $axis attr-dict `:` functional-type(operands, results)
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
    MrtOptTensor:$bias,
    Mrt_I64ArrayType:$strides,
    Mrt_I64ArrayType:$padding,
    Mrt_I64ArrayType:$dilation,
    Mrt_BooleanType:$has_bias
  );

  let results = (outs MrtAnyTensor:$result);

  let assemblyFormat = [{
    $input `,` $kernel (`,` $bias^)? `,` $strides `,` $padding `,` $dilation `,` $has_bias
    attr-dict `:` functional-type(operands, results)
  }];
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

  let generated = 1;
}

def Mrt_MoeGatingTopKSoftmaxOp : Mrt_Op<"moe_gating_top_k_softmax", [Pure]> {
  let summary = "moe calc for input do Softmax, and do topK selection";
  let description = [{
    Performs a Softmax and TopK on input tensors.
  }];
  
  let arguments = (ins
    MrtAnyTensor:$x,
    MrtOptTensor:$finished,
    Mrt_I64Type:$k
  );
  
  let results = (outs 
    MrtAnyTensor:$y,
    MrtAnyTensor:$expert_idx,
    MrtAnyTensor:$row_idx
  );
  
  let assemblyFormat = [{
    $x (`,` $finished^)? `,` $k
    attr-dict `:` functional-type(operands, results)
  }];

  let generated = 1;
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
    MrtAnyTensor:$scaleOptional,
    MrtAnyTensor:$offsetOptional,
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

  let generated = 1;
}

def Mrt_MoeTokenUnpermuteOp : Mrt_Op<"moe_token_unpermute"> {
  let summary = "aclnnMoeTokenUnpermute";
  let description = [{
    Operation for moe token unpermute computation.
    Accepts permuted_tokens, sorted_indices and optional probs tensor, optional padded_mode, optional restore_shape.
    Produces unpermute_tokens as result.
  }];

  let arguments = (ins
    MrtAnyTensor:$permuted_tokens,
    MrtAnyTensor:$sorted_indices,
    MrtOptTensor:$probs,
    Mrt_BooleanType:$padded_mode,
    Mrt_I64ArrayType:$restore_shape
  );

  let results = (outs
    AnyRankedTensor:$Out
  );

  let assemblyFormat = [{
    $permuted_tokens `,` $sorted_indices (`,` $probs^)? `,` $padded_mode `,` $restore_shape
    attr-dict `:` functional-type(operands, results)
  }];
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

  let generated = 1;
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
    Mrt_F32Type:$epsilon
  );

  let results = (outs
    MrtAnyTensor:$yOut,
    MrtAnyTensor:$rstdOut
  );

  let assemblyFormat = [{
    $x `,` $gamma `,` $epsilon
    attr-dict `:` functional-type(operands, results)
  }];

  let generated = 1;
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

def Mrt_SplitOp : Mrt_Op<"split", [Pure]> {
  let summary = "Split operation";
  let description = [{
    Splits tensor along the specified axis into multiple outputs.
  }];

  let arguments = (ins
    MrtAnyTensor:$input,
    Mrt_I64Type:$axis,
    Mrt_I64ArrayType:$split_sizes
  );

  let results = (outs MrtTensorList:$results);

  let assemblyFormat = [{
    $input `,` $axis `,` $split_sizes attr-dict `:` functional-type(operands, results)
  }];
}

def Mrt_SwiGluOp : Mrt_Op<"swi_glu"> {
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

  let generated = 1;
}
#endif // MRT_DIALECT_MRT_OPS_TD
