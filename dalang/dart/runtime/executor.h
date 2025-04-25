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

#include "common/common.h"
#include "tensor/tensor.h"

#include <unordered_map>
#include <vector>

#define DUMP

namespace runtime {
class GraphExecutor {
public:
  GraphExecutor() : context_{tensor::NewDAContext()} { CHECK_IF_NULL(context_); }
  ~GraphExecutor() {
    CHECK_IF_NULL(context_);
    FreeDAContext(context_);
    context_ = nullptr;
  }

  // Start building graph.
  void BeginGraph(const std::string &name);
  // Finish building graph.
  void EndGraph();
  // Add a parameter for graph.
  void AddParameter(tensor::DATensor *param);
  // Add parameters for graph.
  void AddParameters(const std::vector<tensor::DATensor *> &params);
  // Add a const tensor.
  tensor::DATensor *AddTensor(tensor::Type type = tensor::Type_F32, size_t dim = 0,
                      const tensor::ShapeArray &shape = {0}, void *data = nullptr);
  // Add operation result tensor.
  tensor::DATensor *AddTensor(ops::Op op, const std::vector<tensor::DATensor *> &inputs);

  // Run the built graph.
  void RunGraph();
  // If the graph had been built.
  bool HasGraph() const { return graph_ != nullptr; }
#ifdef DUMP
  // Dump the built graph.
  void DumpGraph();
#endif

private:
  void RunTensor(const tensor::DATensor *tensor);

  std::string name_;
  tensor::DAContext *context_{nullptr};
  tensor::DAGraph *graph_{nullptr};
  std::vector<tensor::DATensor *> parameters_;
#ifdef DUMP
  std::unordered_map<const tensor::DATensor *, ssize_t> paraNumMap_;
  std::unordered_map<const tensor::DATensor *, ssize_t> nodeNumMap_;
#endif
};
} // namespace runtime

#endif // __RUNTIME_EXECUTOR_H__