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
#include "ir/pass/pass.h"
#include "ops/operator.h"
#include "runtime/executor/mempool.h"
#include "runtime/utils/utils.h"
#include "ir/tensor/tensor.h"

#define DUMP

namespace da {
namespace runtime {
using tensor::DAContext;
using tensor::DAGraph;
using tensor::DATensor;
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
  void AddParameter(DATensor *param);
  // Add parameters for graph.
  void AddParameters(const std::vector<DATensor *> &params);

  // Add a const tensor.
  DATensor *AddTensor(tensor::Type type = tensor::Type_F32, size_t dim = 0, const tensor::ShapeArray &shape = {0},
                      void *data = nullptr);
  // Add tensor for graph.
  void AddTensor(DATensor *tensor);
  // Add tensor list to tensor.
  void CastToTensorList(DATensor *tensor, size_t len);
  // Add operation result tensor.
  DATensor *AddTensor(ops::Op op, DATensor **start, size_t size);
  DATensor *AddTensor(ops::Op op, const std::vector<DATensor *> &inputs);
  // Add return node.
  DATensor *AddReturn();

  // Run the built graph.
  void RunGraph(bool isDynamic = false);
  // If the graph had been built.
  bool HasGraph() const { return graph_ != nullptr; }
  // Set memory free func for DATensor
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
  void RunTensor(DATensor *tensor);

  std::string name_;
  DAContext *context_{nullptr};
  DAGraph *graph_{nullptr};
  std::vector<DATensor *> parameters_;
  bool isDynamic_{false};
  std::unordered_map<tensor::DATensor *, ops::DAKernel *> kernels_;
  TensorDataRecycler *recycler_{nullptr};
#ifdef DUMP
  std::unordered_map<const DATensor *, ssize_t> paraNumMap_;
  std::unordered_map<const DATensor *, ssize_t> nodeNumMap_;
#endif
};
}  // namespace runtime
}  // namespace da

#endif  // __RUNTIME_EXECUTOR_H__
