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

  let dependentDialects = [
    "::mlir::stablehlo::StablehloDialect",
    "::mlir::func::FuncDialect"
  ];
}

#endif // MOPT_FUSION_PASSES

