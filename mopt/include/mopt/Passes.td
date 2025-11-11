//===- Passes.td - MRT passes definition file ------------*--- tablegen -*-===//
//
// Copyright 2025 Huawei Technologies Co., Ltd
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//

#ifndef MOPT_PASSES
#define MOPT_PASSES

include "mlir/Pass/PassBase.td"

def ConvertReshapeToMRT : Pass<"convert-reshape-to-mrt", "::mlir::ModuleOp"> {
  let summary = "Convert reshape operations to MRT reshape operations";
  let description = [{
    This pass converts reshape operations from other dialects (such as StableHLO)
    to the MRT dialect's reshape operation.
    
    Example:
    ```mlir
    func.func @test(%input: tensor<2x3x4xf32>) -> tensor<24xf32> {
      %0 = "stablehlo.reshape"(%input) : (tensor<2x3x4xf32>) -> tensor<24xf32>
      return %0 : tensor<24xf32>
    }
    ```
    
    After this pass:
    ```mlir
    func.func @test(%input: tensor<2x3x4xf32>) -> tensor<24xf32> {
      %0 = mrt.reshape %input, %shape : (tensor<2x3x4xf32>, tensor<1xi64>) -> tensor<24xf32>
      return %0 : tensor<24xf32>
    }
    ```
  }];
  let constructor = "mlir::createConvertReshapeToMRTPass()";
}

#endif // MOPT_PASSES

