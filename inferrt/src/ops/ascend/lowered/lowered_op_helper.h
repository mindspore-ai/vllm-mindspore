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

#ifndef __OPS_ASCEND_LOWERED_LOWERED_OP_HELPER_H__
#define __OPS_ASCEND_LOWERED_LOWERED_OP_HELPER_H__

#include <string>
#include <memory>
#include "common/visible.h"
#include "ops/operator.h"

namespace mrt::ops {

/**
 * @brief Simplified API for creating lowered operators from MLIR
 *
 * This helper class provides a clean interface for external custom operators,
 * hiding all internal implementation details (AutoLoweredOp, KernelSpec, etc.).
 *
 * External custom ops only need to include this single header file.
 */
class MRT_EXPORT LoweredOpHelper {
 public:
  /**
   * @brief Create an operator from MLIR text
   *
   * This function handles all internal complexity:
   * - Uses mlir_text hash as unique identifier (auto caching)
   * - Registers kernel spec to KernelRegistry
   * - Creates and returns AutoLoweredOp instance
   *
   * Same mlir_text will reuse cached compiled kernel automatically.
   *
   * @param mlir_text MLIR code as string
   * @return Unique pointer to Operator instance, or nullptr on failure
   *
   * Example:
   *   std::string mlir = ReadFile("bias_add.mlir");
   *   auto op = LoweredOpHelper::CreateFromMlirText(mlir);
   */
  static std::unique_ptr<Operator> CreateFromMlirText(const std::string &mlir_text);
};

}  // namespace mrt::ops

#endif  // __OPS_ASCEND_LOWERED_LOWERED_OP_HELPER_H__
