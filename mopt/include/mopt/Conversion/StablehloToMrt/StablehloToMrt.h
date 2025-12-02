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

#ifndef MOPT_CONVERSION_STABLEHLO_TO_MRT_H
#define MOPT_CONVERSION_STABLEHLO_TO_MRT_H

#include <memory>

namespace mlir {
class Pass;

/// Create a pass to convert StableHLO operations to MRT dialect operations.
std::unique_ptr<Pass> createConvertStablehloToMRTPass();

/// Create a pass to convert outlined fusion calls to mrt.linalg_call.
std::unique_ptr<Pass> createConvertOutlinedFusionCallPass();

}  // namespace mlir

#endif  // MOPT_CONVERSION_STABLEHLO_TO_MRT_H

