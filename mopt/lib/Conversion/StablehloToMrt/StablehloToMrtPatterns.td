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

#ifndef STABLEHLO_TO_MRT_PATTERNS
#define STABLEHLO_TO_MRT_PATTERNS

include "mlir/IR/PatternBase.td"
include "stablehlo/dialect/StablehloOps.td"
include "mopt/Dialect/Mrt/Mrt.td"

//===----------------------------------------------------------------------===//
// Simple one-to-one operation mappings
//===----------------------------------------------------------------------===//

// Convert stablehlo.multiply to mrt.mul
def ConvertMulOpPattern : Pat<
  (StableHLO_MulOp $lhs, $rhs),
  (Mrt_MulOp $lhs, $rhs)
>;

// Convert stablehlo.logistic to mrt.sigmoid
def ConvertLogisticOpPattern : Pat<
  (StableHLO_LogisticOp $input, $precision_config),
  (Mrt_SigmoidOp $input)
>;

// Convert stablehlo.dot to mrt.matmul
// Note: This is a simplified pattern. For more complex cases with rank checks,
// you'll still need C++ patterns
def ConvertDotOpPattern : Pat<
  (StableHLO_DotOp:$res $lhs, $rhs, $precision_config),
  (Mrt_MatmulOp $lhs, $rhs),
  [(AnyRankedTensor $res)]
>;

#endif // STABLEHLO_TO_MRT_PATTERNS

