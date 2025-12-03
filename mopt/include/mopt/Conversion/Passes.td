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

#endif // MOPT_CONVERSION_PASSES
