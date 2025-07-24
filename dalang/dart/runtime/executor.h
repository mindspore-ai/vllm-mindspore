/**
 * Copyright 2025 Zhang Qinghua
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

#include <functional>
#include <unordered_map>
#include <vector>

#include "common/common.h"
#include "common/visible.h"
#include "runtime/kernel_lib.h"
#include "runtime/mempool.h"
#include "tensor/tensor.h"

#define DUMP

namespace da {
namespace runtime {
class DA_API GraphExecutor {
public:
  GraphExecutor();
  ~GraphExecutor();

  // Start building graph.
  void BeginGraph(const std::string &name);
  // Finish building graph.
  void EndGraph();
  // Add a parameter for graph.
  void AddParameter(tensor::DATensor *param);
  // Add parameters for graph.
  void AddParameters(const std::vector<tensor::DATensor *> &params);

  // Add a const tensor.
  tensor::DATensor *AddTensor(tensor::Type type = tensor::Type_F32,
                              size_t dim = 0,
                              const tensor::ShapeArray &shape = {0},
                              void *data = nullptr);
  // Add tensor for graph.
  void AddTensor(tensor::DATensor *tensor);
  // Add tensor list to tensor.
  void AddTensorList(tensor::DATensor *tensor, size_t len);
  // Add operation result tensor.
  tensor::DATensor *AddTensor(ops::Op op,
                              const std::vector<tensor::DATensor *> &inputs);

  // Run the built graph.
  void RunGraph();
  // If the graph had been built.
  bool HasGraph() const { return graph_ != nullptr; }
  // Set memory free func for DATensor
  void SetFreeFunc(std::function<void(void *)> &&func) {
    memPool_->SetFreeFunc(std::move(func));
  }
#ifdef DUMP
  // Dump the built graph.
  void DumpGraph();
#endif

private:
  void RunTensor(tensor::DATensor *tensor);

  std::string name_;
  tensor::DAContext *context_{nullptr};
  tensor::DAGraph *graph_{nullptr};
  std::vector<tensor::DATensor *> parameters_;
  runtime::MemoryPool *memPool_{nullptr};
  std::mutex outputsMutex_;
#ifdef DUMP
  std::unordered_map<const tensor::DATensor *, ssize_t> paraNumMap_;
  std::unordered_map<const tensor::DATensor *, ssize_t> nodeNumMap_;
#endif
};
} // namespace runtime
} // namespace da

#endif // __RUNTIME_EXECUTOR_H__