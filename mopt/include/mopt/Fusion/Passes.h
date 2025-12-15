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

#ifndef MOPT_FUSION_PASSES_H
#define MOPT_FUSION_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {

// Forward declarations
class Pass;

//===----------------------------------------------------------------------===//
// Pass declarations
//===----------------------------------------------------------------------===//

#define GEN_PASS_DECL
#include "mopt/Fusion/Passes.h.inc"

//===----------------------------------------------------------------------===//
// Pass creators
//===----------------------------------------------------------------------===//

/// Create a pass to outline StableHLO fusion regions into separate functions.
std::unique_ptr<Pass> createOutlineStablehloFusionRegionsPass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering fusion passes.
#define GEN_PASS_REGISTRATION
#include "mopt/Fusion/Passes.h.inc"  // NOLINT(build/include)

}  // namespace mlir

#endif  // MOPT_FUSION_PASSES_H
