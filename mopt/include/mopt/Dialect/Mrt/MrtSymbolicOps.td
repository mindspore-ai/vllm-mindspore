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

#ifndef MOPT_DIALECT_MRT_MRTSYMBOLICOPS
#define MOPT_DIALECT_MRT_MRTSYMBOLICOPS

include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinAttributes.td"
include "mopt/Dialect/Mrt/MrtDialect.td"
include "mopt/Dialect/Mrt/MrtTypes.td"

def Mrt_SymbolicIntOp : Mrt_Op<"symbolic_int", [Pure]> {
  let summary = "Symbolic int representing a dynamic dimension";
  let description = [{
    The `mrt.symbolic_int` is a one-to-one mapping op of `torch.symbolic_int`.

    Example:
    ```
    %0 = mrt.symbolic_int "s0" {min_val = 5, max_val = 10} : !mrt.164
    %1 = mrt.symbolic_int "s1" {min_val = 2, max_val = 20} : !mrt.164
    ```
  }];
  let arguments = (ins
    StrAttr:$symbol_name,
    I64Attr:$min_val,
    I64Attr:$max_val
  );
  let results = (outs
    Mrt_I64Type:$result
  );
  let assemblyFormat = [{
    $symbol_name ` ` `{` `min_val` `=` $min_val `,` `max_val` `=` $max_val `}` attr-dict `:` type($result)
  }];
}

def Mrt_BindSymbolicShapeOp : Mrt_Op<"bind_symbolic_shape"> {
  let summary = "Binds shape expressions to tensors using an affine map indexed by shape symbols";
  let description = [{
    The `mrt.bind_symbolic_shape` is a one-to-one mapping op of `torch.bind_symbolic_shape`.

    Example:
    ```
    mrt.bind_symbolic_shape %arg0, [%0, %1], affine_map<()[s0, s1] -> (s0, s1, 3)> : !mrt.tensor<?x?x3xf32>
    mrt.bind_symbolic_shape %out0, [%0, %1, %2], affine_map<()[s0, s1, s2] -> (s0, s1 * 2 + s2, 3)> : !mrt.tensor<?x?x3xf32>
    ```
  }];
  let arguments = (ins
    MrtAnyTensor:$operand,
    Variadic<Mrt_I64Type>:$shape_symbols,
    Builtin_AffineMapAttr:$shape_expressions
  );
  let results = (outs);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

#endif // MOPT_DIALECT_MRT_MRTSYMBOLICOPS
