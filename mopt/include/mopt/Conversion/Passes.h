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

#ifndef MOPT_CONVERSION_PASSES_H
#define MOPT_CONVERSION_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {

// Forward declarations
class Pass;

//===----------------------------------------------------------------------===//
// Pass declarations
//===----------------------------------------------------------------------===//

#define GEN_PASS_DECL
#include "mopt/Conversion/Passes.h.inc"

//===----------------------------------------------------------------------===//
// Pass creators
//===----------------------------------------------------------------------===//

/// Create a pass to convert Arith constant operations to MRT constant operations.
std::unique_ptr<Pass> createConvertArithToMRTPass();

/// Create a pass to convert StableHLO operations to MRT dialect operations.
std::unique_ptr<Pass> createConvertStablehloToMRTPass();

/// Create a pass to convert outlined fusion calls to mrt.linalg_call with serialized Linalg MLIR.
std::unique_ptr<Pass> createConvertOutlinedFusionCallPass();

/// Create a pass to convert Torch operations to MRT dialect operations.
std::unique_ptr<Pass> createConvertTorchToMRTPass();

/// Create a pass to convert StableHLO operations to DVM dialect operations.
std::unique_ptr<Pass> createConvertStablehloToDvmPass();

/// Create a pass to serialize DVM dialect graphs to JSON and replace with mrt.dvm_call.
std::unique_ptr<Pass> createConvertDvmToMrtDvmCallPass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering conversion passes.
#define GEN_PASS_REGISTRATION
#include "mopt/Conversion/Passes.h.inc"  // NOLINT(build/include)

}  // namespace mlir

#endif  // MOPT_CONVERSION_PASSES_H
