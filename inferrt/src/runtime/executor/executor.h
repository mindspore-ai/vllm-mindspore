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
#include "runtime/utils/utils.h"
#include "optimize/pass/pass.h"
#include "ir/graph.h"

#define DUMP

namespace mrt {
namespace runtime {
class DA_API GraphExecutor {
 public:
  GraphExecutor();
  ~GraphExecutor();

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

  // Add a const tensor.
  ir::NodePtr AddTensor();
  // Add operation result tensor.
  ir::NodePtr AddTensor(ops::Op op, const std::vector<ir::NodePtr> &inputs);
  // Add return node.
  ir::NodePtr AddReturn();

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
#ifdef DUMP
  std::unordered_map<ir::NodePtr, size_t> paraNumMap_;
  std::unordered_map<ir::NodePtr, size_t> nodeNumMap_;
#endif
};
}  // namespace runtime
}  // namespace mrt

#endif  // __RUNTIME_EXECUTOR_H__
