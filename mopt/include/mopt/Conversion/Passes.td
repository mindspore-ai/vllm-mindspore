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

#ifndef MOPT_CONVERSION_PASSES
#define MOPT_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// StablehloToMrt conversion pass
//===----------------------------------------------------------------------===//

def ConvertStablehloToMRT : Pass<"convert-stablehlo-to-mrt", "::mlir::ModuleOp"> {
  let summary = "Convert StableHLO operations to MRT dialect operations";

  let description = [{
    This pass converts StableHLO dialect operations to the MRT dialect.
    It uses pattern rewriting to match StableHLO operations and
    replace them with equivalent MRT operations, including proper attribute
    mapping and type conversion.
  }];

  let constructor = "mlir::createConvertStablehloToMRTPass()";

  let dependentDialects = [
    "::mlir::stablehlo::StablehloDialect",
    "::mrt::MrtDialect",
    "::mlir::arith::ArithDialect"
  ];
}

def ConvertTorchToMRT : Pass<"convert-torch-to-mrt", "::mlir::ModuleOp"> {
  let summary = "Convert Torch operations to MRT dialect operations";

  let description = [{
    This pass converts Torch dialect operations to the MRT dialect.
  }];

  let constructor = "mlir::createConvertTorchToMRTPass()";

  let dependentDialects = [
    "::mlir::torch::TorchDialect",
    "::mrt::MrtDialect"
  ];
}

//===----------------------------------------------------------------------===//
// ConvertOutlinedFusionCall pass
//===----------------------------------------------------------------------===//

def ConvertOutlinedFusionCall : Pass<"convert-outlined-fusion-call", "::mlir::ModuleOp"> {
  let summary = "Serialize outlined fusion functions to Linalg MLIR text and annotate call sites";

  let description = [{
    This pass processes outlined fusion functions (marked with {fusion.outlined}
    attribute) and their call sites:

    1. Convert StableHLO operations in outlined functions to Linalg dialect
    2. Add hacc attributes (hacc.entry, hacc.function_kind<HOST>) for BiSheng compiler
    3. Serialize the Linalg function to MLIR text format
    4. Attach the serialized MLIR text to corresponding func.call ops as an attribute
       (mrt.outlined_fusion_mlir_text)

    This pass should run after {outline-stablehlo-fusion-regions} and before
    {convert-stablehlo-to-mrt}. The actual replacement of func.call with
    mrt.linalg_call is performed during {convert-stablehlo-to-mrt}, where types
    are converted to MRT tensor types.
  }];

  let constructor = "mlir::createConvertOutlinedFusionCallPass()";

  let dependentDialects = [
    "::mlir::stablehlo::StablehloDialect",
    "::mlir::func::FuncDialect",
    "::mlir::linalg::LinalgDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::math::MathDialect",
    "::mlir::tensor::TensorDialect"
  ];
}

//===----------------------------------------------------------------------===//
// StablehloToDvm conversion pass
//===----------------------------------------------------------------------===//

def ConvertStablehloToDvm : Pass<"convert-stablehlo-to-dvm", "::mlir::ModuleOp"> {
  let summary = "Convert StableHLO operations to DVM dialect operations";

  let description = [{
    This pass converts StableHLO dialect operations to the DVM dialect.
  }];

  let constructor = "mlir::createConvertStablehloToDvmPass()";

  let dependentDialects = [
    "::mlir::stablehlo::StablehloDialect",
    "::mlir::dvm::DvmDialect",
    "::mlir::func::FuncDialect"
  ];
}

#endif // MOPT_CONVERSION_PASSES
