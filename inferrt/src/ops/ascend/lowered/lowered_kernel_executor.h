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

#ifndef __OPS_ASCEND_LOWERED_LOWERED_KERNEL_EXECUTOR_H__
#define __OPS_ASCEND_LOWERED_LOWERED_KERNEL_EXECUTOR_H__

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <functional>

#include "ops/ascend/lowered/kernel_spec.h"
#include "ir/graph.h"

namespace mrt::ops {

// Forward declarations
class LoweredKernelCacheEntry;
using LoweredCacheEntryPtr = std::unique_ptr<LoweredKernelCacheEntry>;

/**
 * @brief Host API function signature
 * typedef void (*host_api_function_t)(uint32_t blockDim, void* l2ctrl, void* stream, void** args)
 */
using HostApiFunction = std::function<void(uint32_t, void *, void *, void **)>;

/**
 * @brief Tiling function signatures for dynamic shape kernels
 */
using GetTilingSizeFunction = std::function<int64_t()>;
using TilingFunction = std::function<void(void **)>;

/**
 * @brief Cache entry for a specific kernel instance
 *
 * Caches dlopen handle, function pointers, tiling data, and blockDim
 * for reuse across multiple kernel launches with the same configuration.
 */
class LoweredKernelCacheEntry {
 public:
  LoweredKernelCacheEntry() = default;
  ~LoweredKernelCacheEntry();

  // Disable copy, allow move
  LoweredKernelCacheEntry(const LoweredKernelCacheEntry &) = delete;
  LoweredKernelCacheEntry &operator=(const LoweredKernelCacheEntry &) = delete;
  LoweredKernelCacheEntry(LoweredKernelCacheEntry &&) = default;
  LoweredKernelCacheEntry &operator=(LoweredKernelCacheEntry &&) = default;

  void *dlHandle = nullptr;             // dlopen handle
  HostApiFunction hostApiFunc;          // Host API function pointer
  GetTilingSizeFunction getTilingSize;  // Tiling size function (optional)
  TilingFunction tilingFunc;            // Tiling function (optional)

  uint32_t blockDim = 0;            // Cached block dimension
  size_t workspaceSize = 0;         // Cached workspace size
  int64_t tilingStructSize = 0;     // Size of tiling data structure
  std::vector<int64_t> tilingData;  // Cached tiling data (host-side)
  void *dTilingData = nullptr;      // Device-side tiling data buffer
  int64_t tilingKey = 0;            // Tiling key for this entry
};

/**
 * @brief Executor for lowered kernels using Host API
 *
 * This class manages the execution of kernels compiled from BiSheng IR.
 * It handles:
 * - Dynamic library loading (dlopen/dlsym)
 * - Tiling computation for dynamic shape kernels
 * - Argument marshaling (memref structures)
 * - Caching for repeated executions
 * - Workspace size calculation
 *
 * The executor follows a similar pattern to AclnnExecutor but uses Host API
 * instead of ACLNN API.
 */
class LoweredKernelExecutor {
 public:
  /**
   * @brief Construct an executor for a specific kernel
   * @param spec Pointer to KernelSpec (non-owning, must outlive this executor)
   */
  explicit LoweredKernelExecutor(const KernelSpec *spec);

  ~LoweredKernelExecutor();

  /**
   * @brief Calculate workspace size required for kernel execution
   *
   * For dynamic shape kernels, this will:
   * 1. Look up cache based on input shapes
   * 2. If miss, load kernel and compute tiling
   * 3. Return workspace size (tiling data size)
   *
   * @param workspaceSize Output: required workspace size in bytes
   * @param inputs Vector of input Values (containing shape information)
   * @param output Output Value (containing shape information)
   * @return 0 on success, error code otherwise
   */
  int GetWorkspaceSize(size_t *workspaceSize, const std::vector<const ir::Value *> &inputs, const ir::Value *output);

  /**
   * @brief Launch the kernel on the device
   *
   * Uses the cached configuration (from GetWorkspaceSize) to execute the kernel.
   *
   * @param workspace Device memory workspace buffer
   * @param workspaceSize Size of workspace buffer
   * @param stream ACL runtime stream
   * @param inputs Vector of input Values (containing device pointers)
   * @param output Output Value (containing device pointer)
   * @return 0 on success, error code otherwise
   */
  int Launch(void *workspace, size_t workspaceSize, void *stream, const std::vector<const ir::Value *> &inputs,
             const ir::Value *output);

  /**
   * @brief Get the kernel specification
   * @return Pointer to KernelSpec, or nullptr if not found
   */
  const KernelSpec *GetSpec() const { return spec_; }

 private:
  /**
   * @brief Generate cache key based on shapes
   */
  std::string GenerateCacheKey(const std::vector<const ir::Value *> &inputs, const ir::Value *output) const;

  /**
   * @brief Load kernel library and symbols
   * @return LoweredKernelCacheEntry with loaded functions, or nullptr on failure
   */
  LoweredCacheEntryPtr LoadKernel();

  /**
   * @brief Compute tiling parameters for dynamic shape kernels
   *
   * @param inputs Input values
   * @param output Output value
   * @param entry Cache entry to store tiling results
   * @return 0 on success, error code otherwise
   */
  int ComputeTiling(const std::vector<const ir::Value *> &inputs, const ir::Value *output,
                    LoweredKernelCacheEntry *entry);

  /**
   * @brief Build kernel arguments (memref structures + tiling)
   * @param inputs Input values
   * @param output Output value
   * @param entry Cache entry (may contain tiling data)
   * @param args Output: vector of argument pointers
   */
  void BuildKernelArgs(const std::vector<const ir::Value *> &inputs, const ir::Value *output,
                       const LoweredKernelCacheEntry *entry, std::vector<void *> *args);

  /**
   * @brief Add memref arguments for N-dimensional tensor
   * Format: {allocated_ptr, aligned_ptr, offset, sizes[N], strides[N]}
   * @param ptr Device pointer to tensor data
   * @param shape Tensor shape (dimensions)
   * @param args Output argument vector
   */
  void AddMemrefArgs(void *ptr, const std::vector<int64_t> &shape, std::vector<void *> *args);

  /**
   * @brief Add tiling arguments to kernel args
   * @param entry Cache entry (may contain tiling data)
   * @param args Output argument vector
   */
  void AddTilingArgs(const LoweredKernelCacheEntry *entry, std::vector<void *> *args);

  const KernelSpec *spec_;  // Non-owning pointer to spec

  std::string cacheDir_;           // Cache directory for compiled kernel (for cleanup)
  bool keepIntermediateFiles_;     // Whether to keep intermediate files for debugging

  std::unordered_map<std::string, LoweredCacheEntryPtr> cache_;  // Cache by shape+config key

  LoweredKernelCacheEntry *currentEntry_;  // Current cache entry for Launch reuse (non-owning)
};

}  // namespace mrt::ops

#endif  // __OPS_ASCEND_LOWERED_LOWERED_KERNEL_EXECUTOR_H__
