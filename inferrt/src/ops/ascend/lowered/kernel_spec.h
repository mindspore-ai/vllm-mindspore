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
#include <mutex>
#include <functional>

#include "common/visible.h"

namespace mrt::ops {

/**
 * @brief Compiler callback function signature
 *
 * Compiles MLIR text to a .so file and returns the .so path.
 * @param mlirText MLIR text content
 * @param outputSoPath Output path for compiled .so file (suggested, can be modified)
 * @param entryName Output entry function name (will be filled by compiler)
 * @param tilingPrefix Output tiling function prefix (will be filled by compiler)
 * @return true on success, false on failure
 */
using MlirCompilerCallback = std::function<bool(const std::string &mlirText, std::string &outputSoPath,
                                                std::string &entryName, std::string &tilingPrefix)>;

/**
 * @brief Kernel specification describing how to load and execute a lowered kernel
 *
 * This struct contains all necessary metadata to load a lowered kernel compiled
 * from MLIR IR and execute it via Host API.
 *
 * Supports dynamic compilation: mlirText is compiled on first use, with caching
 * based on MLIR text hash to avoid recompilation.
 */
struct KernelSpec {
  std::string id;  // Unique identifier for this kernel spec (e.g., "bias_add")

  // --- Dynamic MLIR compilation mode ---
  std::string mlirText;          // MLIR text to compile
  MlirCompilerCallback compiler;  // Compiler callback for MLIR → .so

  // --- Compiled kernel metadata (populated after compilation) ---
  std::string kernelLibPath;  // Path to the compiled kernel .so file
  std::string entry;            // Host API function entry point name
  std::string tilingPrefix;    // Prefix for tiling-related functions (for dynamic shape)

  mutable std::mutex compilationMutex_;  // Protect compilation state in multithreaded environment

  KernelSpec(const std::string &id_, const std::string &mlirText_);

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

/**
 * @brief Registry for managing kernel specifications
 *
 * This singleton class maintains a registry of all custom kernel specifications.
 * It allows registering new kernels and looking up their specifications by ID.
 *
 * Thread-safe for concurrent registration and lookup.
 */
class KernelRegistry {
 public:
  /**
   * @brief Get the singleton instance
   * @return Reference to the global KernelRegistry instance
   */
  static KernelRegistry &Instance();

  /**
   * @brief Register a kernel specification
   * @param specId Unique identifier for the kernel
   * @param mlirText MLIR text for the kernel
   * @return true if registration successful, false if specId already exists
   */
  bool Register(const std::string &specId, const std::string &mlirText);

  /**
   * @brief Look up a kernel specification by ID
   * @param specId Identifier to look up
   * @return Pointer to KernelSpec if found, nullptr otherwise
   */
  const KernelSpec *Lookup(const std::string &specId) const;

  /**
   * @brief Check if a kernel spec is registered
   * @param specId Identifier to check
   * @return true if registered, false otherwise
   */
  bool Contains(const std::string &specId) const;

  // Disable copy and move
  KernelRegistry(const KernelRegistry &) = delete;
  KernelRegistry &operator=(const KernelRegistry &) = delete;
  KernelRegistry(KernelRegistry &&) = delete;
  KernelRegistry &operator=(KernelRegistry &&) = delete;

 private:
  KernelRegistry() = default;
  ~KernelRegistry() = default;

  mutable std::mutex mutex_;
  std::unordered_map<std::string, KernelSpec> specs_;
};

}  // namespace mrt::ops

#endif  // __OPS_ASCEND_LOWERED_KERNEL_SPEC_H__
