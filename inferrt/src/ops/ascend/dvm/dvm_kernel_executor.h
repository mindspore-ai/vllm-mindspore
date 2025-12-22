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

#ifndef __OPS_ASCEND_DVM_DVM_KERNEL_EXECUTOR_H__
#define __OPS_ASCEND_DVM_DVM_KERNEL_EXECUTOR_H__

#include <string>
#include <vector>
#include <memory>
#include <functional>

#include "ops/ascend/dvm/prebuild/dvm.h"
#include "ir/graph.h"

namespace mrt::ops {

/**
 * @brief Executor for DVM kernels
 *
 * Manages the execution of DVM-based kernels. Handles:
 * - Kernel graph construction using DVM API
 * - Shape lifecycle management (ShapeRef pattern)
 * - RelocTable building for runtime address mapping
 * - CodeGen and Launch orchestration
 * - Caching for static shape kernels
 *
 * Similar to LoweredKernelExecutor but uses DVM API instead of Host API.
 */
class DvmKernelExecutor {
 public:
  /**
   * @brief Construct executor for specific kernel type
   * @param kernelType Type of DVM kernel (static/dynamic shape)
   */
  explicit DvmKernelExecutor(dvm::KernelType kernelType);

  ~DvmKernelExecutor();

  /**
   * @brief Build computation graph using DVM API
   *
   * Calls user-provided build function to construct the graph:
   * - kernel_.Load() for inputs -> save returned NDObject*
   * - kernel_.Binary/Unary/MatMul/etc for operations
   * - kernel_.Store() for outputs -> save returned NDObject*
   *
   * The buildFunc must populate inputObjs and outputObjs with NDObject* pointers
   * returned from Load() and Store() calls. These are required for RelocTable.
   *
   * @param buildFunc Function that builds graph and records NDObject pointers
   * @param inputs Input values (for shape information during build)
   * @param output Output value (for shape information during build)
   * @return 0 on success, error code otherwise
   */
  int BuildKernel(std::function<void(dvm::Kernel &, const std::vector<const ir::Value *> &, const ir::Value *,
                                     const std::vector<dvm::ShapeRef *> &, const std::vector<dvm::ShapeRef *> &,
                                     std::vector<dvm::NDObject *> *, std::vector<dvm::NDObject *> *)>
                    buildFunc,
                  const std::vector<const ir::Value *> &inputs, const ir::Value *output);

  /**
   * @brief Get workspace size required for kernel execution
   *
   * For dynamic shape kernels:
   * - Updates input shape refs
   * - Calls kernel_.CodeGen()
   * - Returns workspace size
   *
   * For static shape kernels:
   * - CodeGen called once, result cached
   *
   * @param workspaceSize Output: required workspace size in bytes
   * @param inputs Input values (containing shape information)
   * @param output Output value (containing shape information)
   * @return 0 on success, error code otherwise
   */
  int GetWorkspaceSize(size_t *workspaceSize, const std::vector<const ir::Value *> &inputs, const ir::Value *output);

  /**
   * @brief Launch the kernel on device
   *
   * Workflow:
   * 1. Build RelocTable mapping NDObject* to device addresses
   * 2. Call kernel_.Launch(reloc_table, inputs, outputs, workspace, stream)
   *
   * @param workspace Device memory workspace buffer
   * @param workspaceSize Size of workspace buffer
   * @param stream ACL runtime stream
   * @param inputs Input values (containing device pointers)
   * @param output Output value (containing device pointer)
   * @return 0 on success, error code otherwise
   */
  int Launch(void *workspace, size_t workspaceSize, void *stream, const std::vector<const ir::Value *> &inputs,
             ir::Value *output);

 private:
  /**
   * @brief Update shape references from ir::Value tensors
   *
   * Copies shape data into shapesStorage_ and creates ShapeRef objects
   * pointing to the stored data. Ensures shape data remains valid during
   * kernel execution.
   *
   * @param inputs Input values
   * @param output Output value
   */
  void UpdateShapeRefs(const std::vector<const ir::Value *> &inputs, const ir::Value *output);

  void EnsureShapeRefsInitialized(size_t numInputs, size_t numOutputs);

  /**
   * @brief Build relocation table for kernel launch
   *
   * Maps NDObject pointers to actual device addresses from ir::Value tensors.
   *
   * @param inputs Input values (containing device pointers)
   * @param output Output value (containing device pointer)
   */
  void BuildRelocTable(const std::vector<const ir::Value *> &inputs, ir::Value *output);

  dvm::KernelType kernelType_;  // Kernel type (static/dynamic shape)
  dvm::Kernel kernel_;          // DVM kernel instance

  // Shape lifecycle management
  // Pattern: shapesStorage_ owns the data, shapeRefs_ holds lightweight references
  std::vector<std::vector<int64_t>> shapesStorage_;  // Owns shape data
  std::vector<dvm::ShapeRef> shapeRefs_;             // References to storage (addresses must stay stable)
  std::vector<dvm::ShapeRef *> inputShapeRefPtrs_;   // Persistent pointers to input ShapeRefs
  std::vector<dvm::ShapeRef *> outputShapeRefPtrs_;  // Persistent pointers to output ShapeRefs

  // RelocTable for Launch
  dvm::RelocTable relocTable_;
  std::vector<dvm::NDObject *> inputNDObjects_;   // Input NDObjects from Load()
  std::vector<dvm::NDObject *> outputNDObjects_;  // Output NDObjects from Store()

  // CodeGen cache for static shape kernels
  bool isCodeGenDone_;
  uint64_t cachedWorkspaceSize_;

  // Build state
  bool isKernelBuilt_;
};

}  // namespace mrt::ops

#endif  // __OPS_ASCEND_DVM_DVM_KERNEL_EXECUTOR_H__
