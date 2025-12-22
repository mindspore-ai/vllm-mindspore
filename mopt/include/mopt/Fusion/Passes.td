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

#ifndef MOPT_FUSION_PASSES
#define MOPT_FUSION_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// OutlineStablehloFusionRegions pass
//===----------------------------------------------------------------------===//

def OutlineStablehloFusionRegions : Pass<"outline-stablehlo-fusion-regions", "::mlir::ModuleOp"> {
  let summary = "Outline StableHLO fusion regions into separate functions";

  let description = [{
    This pass identifies fusible StableHLO operations and outlines them into
    separate functions marked with the {fusion.outlined} attribute.

    Fusion roots (starting points):
    - Elementwise operations only: add, subtract, multiply, divide, tanh, exp, etc.

    Note: Compute-intensive ops (dot_general, convolution, reduce) are NOT fusion
    roots - they are handled by the standard MRT lowering path.

    Operations that can be fused into roots:
    - Elementwise: add, subtract, multiply, divide, max, min, tanh, exp, log, etc.
    - Broadcast: broadcast_in_dim
    - Constants: stablehlo.constant

    Operations that are NOT fused:
    - Compute-intensive ops: dot_general, convolution, reduce
    - Memory ops: scatter, gather, sort
    - Collective communication operations
    - Operations with dynamic shapes
    - Intermediate results with multiple external consumers
  }];

  let constructor = "mlir::createOutlineStablehloFusionRegionsPass()";

  let options = [
    Option<"allowDotGeneral", "allow-dot-general", "bool", "false",
           "Allow stablehlo.dot / stablehlo.dot_general (MatMul) to be fused into clusters (as producers)">,
    Option<"allowDynamicShape", "allow-dynamic-shape", "bool", "false",
           "Allow operations with dynamic shapes to be fused into clusters">
  ];

  let dependentDialects = [
    "::mlir::stablehlo::StablehloDialect",
    "::mlir::func::FuncDialect"
  ];
}

//===----------------------------------------------------------------------===//
// MarkTorchToStablehloOp pass
//===----------------------------------------------------------------------===//

def MarkTorchToStablehloOp : Pass<"mark-torch-to-stablehlo-op", "::mlir::ModuleOp"> {
  let summary = "Mark Torch ops to be converted to StableHLO";
  let description = [{
    This pass marks Torch operations that should be converted to StableHLO based on:
    1. An environment variable MOPT_TORCH_TO_STABLEHLO_WHITELIST
    2. Custom C++ matching rules

    Marked operations will have the 'mopt.torch_to_stablehlo' attribute added.
  }];
  let constructor = "mlir::createMarkTorchToStablehloOpPass()";
}

// PropagateTorchSymbolicShape pass
//===----------------------------------------------------------------------===//

def PropagateTorchSymbolicShape : Pass<"propagate-torch-symbolic-shape", "::mlir::func::FuncOp"> {
  let summary = "Propagate torch.bind_symbolic_shape metadata onto builtin tensor producers";
  let description = [{
    TorchDynamo symbolic shapes are modeled with:
      - torch.symbolic_int
      - torch.bind_symbolic_shape

    StableHLO/CHLO lowering typically loses this relationship information and
    may introduce dynamic broadcast chains (shape.shape_of/shape.broadcast/
    stablehlo.dynamic_broadcast_in_dim) even when operands are known (via
    symbolic bindings) to have identical shapes.

    This pass derives a stable, comparable "symbolic shape signature" from
    torch.bind_symbolic_shape and attaches it to:
      - torch_c.to_builtin_tensor results (builtin tensor values)
      - the producer op of a builtin tensor that is immediately wrapped by
        torch_c.from_builtin_tensor and then bound by torch.bind_symbolic_shape

    Downstream passes (e.g. CHLO->StableHLO legalization) can consult this
    metadata to avoid emitting redundant dynamic broadcasts.
  }];

  let constructor = "mlir::createPropagateTorchSymbolicShapePass()";

  let dependentDialects = [
    "::mlir::func::FuncDialect",
    "::mlir::torch::Torch::TorchDialect",
    "::mlir::torch::TorchConversion::TorchConversionDialect"
  ];
}

//===----------------------------------------------------------------------===//
// RemoveRedundantDynamicBroadcast pass
//===----------------------------------------------------------------------===//

def RemoveRedundantDynamicBroadcast : Pass<"remove-redundant-dynamic-broadcast", "::mlir::func::FuncOp"> {
  let summary = "Remove redundant dynamic broadcast chains using symbolic shape information";
  let description = [{
    This pass removes redundant dynamic broadcast operations that were generated
    by CHLO->StableHLO legalization when the operands are known (via mopt.symbolic_shape
    metadata) to have identical shapes.

    In dynamic shape scenarios, CHLO ops are lowered to:
      shape.shape_of + shape.broadcast + shape.cstr_broadcastable +
      shape.assuming { dynamic_broadcast_in_dim + elementwise_op }

    When both operands have the same symbolic shape signature, broadcasting is
    provably a no-op. This pass:
      1. Identifies shape.assuming regions containing dynamic broadcasts
      2. Checks if all dynamic_broadcast_in_dim inputs have matching mopt.symbolic_shape
      3. If so, inlines the assuming region and removes the broadcast ops

    This enables fusion passes to see clean elementwise chains without shape
    dependencies blocking cluster formation.
  }];

  let constructor = "mlir::createRemoveRedundantDynamicBroadcastPass()";

  let dependentDialects = [
    "::mlir::stablehlo::StablehloDialect",
    "::mlir::shape::ShapeDialect",
    "::mlir::func::FuncDialect"
  ];
}

//===----------------------------------------------------------------------===//
// VerifySymbolicShapeAttrs pass
//===----------------------------------------------------------------------===//

def VerifySymbolicShapeAttrs : Pass<"verify-symbolic-shape-attrs", "::mlir::func::FuncOp"> {
  let summary = "Verify mopt.symbolic_shape attribute schema/type";
  let description = [{
    Validates that `mopt.symbolic_shape` is a structured DictionaryAttr with the expected
    schema keys and types. This is a lightweight contract checker useful during debugging
    and refactors of symbolic-shape propagation/cleanup passes.
  }];

  let constructor = "mlir::createVerifySymbolicShapeAttrsPass()";

  let dependentDialects = [
    "::mlir::func::FuncDialect"
  ];
}

#endif // MOPT_FUSION_PASSES
