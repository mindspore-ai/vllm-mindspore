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

#ifndef __OPS_ASCEND_DVM_DVM_OP_H__
#define __OPS_ASCEND_DVM_DVM_OP_H__

#include <string>
#include <memory>
#include <functional>

#include "common/visible.h"
#include "ops/operator.h"
#include "ops/op_register.h"
#include "ops/ascend/dvm/prebuild/dvm.h"
#include "ops/ascend/dvm/dvm_kernel_executor.h"

namespace mrt::ops {

/**
 * @brief DVM operator wrapper
 *
 * Provides Operator interface for DVM-based kernels. Delegates to
 * DvmKernelExecutor for actual kernel management and execution.
 *
 * Usage pattern:
 * ```cpp
 * class MyDvmOp : public DvmOp {
 *  public:
 *   MyDvmOp() : DvmOp(dvm::kStaticShape, BuildMyGraph) {}
 *
 *  private:
 *   static void BuildMyGraph(dvm::Kernel& kernel,
 *                           const std::vector<const ir::Value*>& inputs,
 *                           const ir::Value* output) {
 *     // Build computation graph using DVM API
 *     auto* a = kernel.Load(nullptr, nullptr, dvm::kFloat16);
 *     auto* b = kernel.Load(nullptr, nullptr, dvm::kFloat16);
 *     auto* result = kernel.Binary(dvm::kAdd, a, b);
 *     kernel.Store(nullptr, result);
 *   }
 * };
 * ```
 */
class DvmOp : public Operator {
 public:
  /**
   * @brief Build function signature
   *
   * User provides a function that builds the computation graph:
   * - kernel: DVM kernel instance to build graph on
   * - inputs: Input values (for shape/type information)
   * - output: Output value (for shape/type information)
   * - inputObjs: Output parameter to record NDObject* from Load() calls
   * - outputObjs: Output parameter to record NDObject* from Store() calls
   *
   * IMPORTANT: buildFunc MUST push NDObject* returned by kernel.Load() to inputObjs
   * and NDObject* returned by kernel.Store() to outputObjs. These are required for
   * building the RelocTable used during Launch.
   */
  // NOTE: DVM Kernel::Load takes a ShapeRef*; for dynamic-shape kernels DVM may
  // keep and consult these pointers during CodeGen/Launch. Therefore we plumb
  // persistent ShapeRef pointers owned by DvmKernelExecutor into buildFunc.
  using BuildFunc = std::function<void(dvm::Kernel &, const std::vector<const ir::Value *> &, const ir::Value *,
                                       const std::vector<dvm::ShapeRef *> & /*inputShapeRefs*/,
                                       const dvm::ShapeRef * /*outputShapeRef*/, std::vector<dvm::NDObject *> *,
                                       std::vector<dvm::NDObject *> *)>;

  /**
   * @brief Construct DVM operator
   *
   * @param kernelType Type of DVM kernel (static/dynamic shape)
   * @param buildFunc Function to build computation graph and record NDObjects
   *
   * Example buildFunc implementation:
   * @code
   * [](dvm::Kernel& k, const auto& ins, const auto* out,
   *    std::vector<dvm::NDObject*>* inObjs, std::vector<dvm::NDObject*>* outObjs) {
   *   auto* a = k.Load(nullptr, nullptr, dvm::kFloat16);
   *   auto* b = k.Load(nullptr, nullptr, dvm::kFloat16);
   *   inObjs->push_back(a);  // REQUIRED
   *   inObjs->push_back(b);  // REQUIRED
   *
   *   auto* result = k.Binary(dvm::kAdd, a, b);
   *   auto* stored = k.Store(nullptr, result);
   *   outObjs->push_back(stored);  // REQUIRED
   * }
   * @endcode
   */
  explicit DvmOp(dvm::KernelType kernelType, BuildFunc buildFunc);

  ~DvmOp() override = default;

  /**
   * @brief Initialize operator (optional override)
   *
   * Can be used to perform one-time setup with actual input/output values.
   *
   * @param inputs Input values
   * @param output Output value
   */
  void Init(const std::vector<const ir::Value *> &inputs, const ir::Value *output) override;

  /**
   * @brief Calculate workspace size
   *
   * For dynamic shape kernels:
   * - Updates input shapes in executor
   * - Calls kernel CodeGen
   * - Returns workspace size
   *
   * For static shape kernels:
   * - CodeGen called once, result cached
   *
   * @param input Input values
   * @param output Output value
   * @param workspaceSize Output: workspace size in bytes
   * @return SUCCESS on success, error code otherwise
   */
  OpsErrorCode CalcWorkspace(const std::vector<const ir::Value *> &input, const ir::Value *output,
                             size_t *workspaceSize) override;

  /**
   * @brief Launch kernel on device
   *
   * Delegates to executor which:
   * 1. Builds RelocTable with actual device addresses
   * 2. Calls kernel_.Launch(reloc_table, inputs, outputs, workspace, stream)
   *
   * @param input Input values
   * @param workspace Workspace buffer
   * @param workspaceSize Workspace buffer size
   * @param output Output value
   * @param stream Execution stream
   * @return SUCCESS on success, error code otherwise
   */
  OpsErrorCode Launch(const std::vector<const ir::Value *> &input, void *workspace, size_t workspaceSize,
                      ir::Value *output, void *stream) override;

 protected:
  dvm::KernelType kernelType_;                   // Kernel type
  BuildFunc buildFunc_;                          // User-provided graph builder
  std::unique_ptr<DvmKernelExecutor> executor_;  // Kernel executor
  bool isInitialized_;                           // Initialization flag
};

}  // namespace mrt::ops

#endif  // __OPS_ASCEND_DVM_DVM_OP_H__
