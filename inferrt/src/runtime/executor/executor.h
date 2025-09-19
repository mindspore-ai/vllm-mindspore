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

#ifndef __RUNTIME_EXECUTOR_H__
#define __RUNTIME_EXECUTOR_H__

#include <utility>
#include <string>
#include <functional>
#include <unordered_map>
#include <vector>

#include "common/common.h"
#include "common/visible.h"
#include "ops/operator.h"
#include "runtime/executor/mempool.h"
#include "runtime/executor/op_runner.h"
#include "runtime/builder/builder.h"
#include "runtime/utils/utils.h"
#include "hardware/hardware_abstract/device_context.h"
#include "optimize/pass/pass.h"
#include "ir/graph.h"

#define DUMP

namespace mrt {
namespace runtime {
enum ExecutionMode : size_t {
  Base = 0,
  Pipeline = 1,
};

/**
 * @brief Base class for executing a computational graph.
 *
 * The Executor class provides the basic interface and implementation for
 * running computational graph. It holds the graph and operation runners
 * needed for execution.
 */
class DA_API Executor {
 public:
  Executor() = delete;
  Executor(const std::shared_ptr<std::vector<OpRunner>> &opRunners,
           const std::map<hardware::DeviceType, device::DeviceContext *> &deviceContexts)
      : opRunners_(opRunners), deviceContexts_(deviceContexts) {}

  virtual ~Executor() = default;

  /**
   * @brief Executes the computational graph.
   * This method runs all operations in the graph by execution order.
   * Subclasses can override this method to provide specialized execution behavior, such as Pipeline mode, AclGraph
   * mode.
   * @param isDynamic whether run graph by dynamic shape mode.
   */
  virtual void Run(bool isDynamic);

 protected:
  // Shared pointer to the vector of OpRunners for all operators by execution order.
  std::shared_ptr<std::vector<OpRunner>> opRunners_{nullptr};

  std::map<hardware::DeviceType, device::DeviceContext *> deviceContexts_{};
};

class DA_API GraphExecutor {
 public:
  GraphExecutor();
  ~GraphExecutor();

  // 1. Graph construct and optimize, mlrl->infer ir->execution order(op runners)
  // Start building graph.
  void BeginGraph(const std::string &name);
  // Finish building graph.
  void EndGraph();
  // Optimize the graph.
  void OptGraph();
  // Build DAKernels for graph.
  void BuildKernels();
  // Add a parameter for graph.
  void AddParameter(ir::NodePtr param);
  // Add a value node.
  ir::NodePtr AddValueNode(const ir::ValuePtr &value = nullptr);
  // Add an operation node.
  ir::NodePtr AddOpNode(ops::Op op, const std::vector<ir::NodePtr> &inputs, const ir::ValuePtr &output = nullptr);
  // Add return node.
  ir::NodePtr AddReturn();

  // 2. Create Builder, analyse execution order and create Executor by execution mode.
  void BuildExecutor();

  // 3. Run graph via Executor.
  // Run the built graph.
  void RunGraph(bool isDynamic = false);
  // If the graph had been built.
  bool HasGraph() const { return graph_ != nullptr; }
  // Set memory free func for Tensor data
  void SetFreeFunc(std::function<void(void *)> &&func) {
    CHECK_IF_NULL(recycler_);
    recycler_->SetFreeFunc(std::move(func));
  }
  // Free the memory of graph outputs
  void FreeGraphOutputs();
  // Record tensor refCount
  void RecordTensorRefCount();
#ifdef DUMP
  // Dump the built graph.
  void DumpGraph();
#endif

 private:
  void RunNode(ir::NodePtr node);

  std::string name_;
  ir::GraphPtr graph_;
  std::vector<ir::NodePtr> parameters_;
  bool isDynamic_{false};
  std::unordered_map<ir::NodePtr, ops::DAKernel *> kernels_;
  TensorDataRecycler *recycler_{nullptr};

  std::unique_ptr<Builder> builder_{nullptr};
  std::unique_ptr<Executor> executor_{nullptr};
#ifdef DUMP
  std::unordered_map<ir::NodePtr, size_t> paraNumMap_;
  std::unordered_map<ir::NodePtr, size_t> nodeNumMap_;
#endif
};
}  // namespace runtime
}  // namespace mrt

#endif  // __RUNTIME_EXECUTOR_H__
