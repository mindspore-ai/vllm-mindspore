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

#ifndef __OPS_ASCEND_LOWERED_KERNEL_SPEC_H__
#define __OPS_ASCEND_LOWERED_KERNEL_SPEC_H__

#include <string>
#include <unordered_map>
#include <memory>
#include <functional>

#include "common/visible.h"

namespace mrt::ops {

/**
 * @brief Kernel specification describing how to load and execute a lowered kernel
 *
 * Pure data structure containing all necessary metadata to load a lowered kernel
 * compiled from MLIR IR and execute it via Host API.
 *
 * Compilation is managed by LoweredKernelExecutor, not by this struct.
 */
struct KernelSpec {
  std::string id;  // Unique identifier for this kernel spec (e.g., "bias_add")

  // --- Dynamic MLIR compilation mode ---
  std::string mlirText;  // MLIR text to compile (if not empty, requires compilation)

  // --- Compiled kernel metadata (populated after compilation) ---
  std::string kernelLibPath;  // Path to the compiled kernel .so file
  std::string entry;          // Host API function entry point name (also used as tiling function prefix)

  explicit KernelSpec(const std::string &mlirText_) : id("pending"), mlirText(mlirText_) {}
  /**
   * @brief Check if this spec needs compilation
   * @return true if kernel hasn't been compiled yet
   */
  bool NeedsCompilation() const { return kernelLibPath.empty(); }

  /**
   * @brief Check if this spec is ready to load (has .so path)
   * @return true if kernelLibPath is available
   */
  bool IsReadyToLoad() const { return !kernelLibPath.empty(); }

  /**
   * @brief Auto-detect if kernel uses dynamic shapes from MLIR text
   *
   * Dynamic shape tensors contain '?' in type signatures (e.g., tensor<?x?xf16>)
   * Static shape tensors have concrete dimensions (e.g., tensor<1x6144xf16>)
   *
   * @return true if MLIR contains dynamic shape markers ('?'), false otherwise
   */
  bool IsDynamicShape() const {
    // Check for dynamic dimensions in both tensor and memref types
    // Examples:
    //   Dynamic: tensor<?x?xf16>, tensor<?x4096xf32>, memref<4x?xf32>
    //   Static:  tensor<1x6144xf16>, tensor<4096xf32>, memref<4x8xf32>

    // Check for "tensor<...?..." or "memref<...?..." patterns
    size_t pos = 0;
    while ((pos = mlirText.find('<', pos)) != std::string::npos) {
      // Check if this '<' is preceded by "tensor" or "memref"
      if (pos >= 6 && mlirText.substr(pos - 6, 6) == "tensor") {
        // Found "tensor<", now check if there's '?' before the closing '>'
        size_t closeBracket = mlirText.find('>', pos);
        if (closeBracket != std::string::npos) {
          if (mlirText.find('?', pos) < closeBracket) {
            return true;
          }
        }
      } else if (pos >= 6 && mlirText.substr(pos - 6, 6) == "memref") {
        // Found "memref<", now check if there's '?' before the closing '>'
        size_t closeBracket = mlirText.find('>', pos);
        if (closeBracket != std::string::npos) {
          if (mlirText.find('?', pos) < closeBracket) {
            return true;
          }
        }
      }
      ++pos;
    }
    return false;
  }
};

}  // namespace mrt::ops

#endif  // __OPS_ASCEND_LOWERED_KERNEL_SPEC_H__
