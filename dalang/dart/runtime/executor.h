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
#include "tensor/da_tensor.h"

#include <unordered_map>
#include <vector>

#define DUMP

namespace runtime {
using namespace tensor;

class GraphExecutor {
public:
  GraphExecutor() : context_{NewDAContext()} { CHECK_IF_NULL(context_); }
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
  void AddParameter(DATensor *param);
  // Add parameters for graph.
  void AddParameters(const std::vector<DATensor *> &params);
  // Add a const tensor.
  DATensor *AddTensor(Type type = Type_F32, size_t dim = 0,
                      const ShapeArrayPtr &shape = {0},
                      void *data = nullptr);
  // Add operation result tensor.
  DATensor *AddTensor(ops::Op op, const std::vector<DATensor *> &inputs);

  // Run the built graph.
  void RunGraph();
  // If the graph had been built.
  bool HasGraph() const { return graph_ != nullptr; }
#ifdef DUMP
  // Dump the built graph.
  void DumpGraph();
#endif

private:
  void RunTensor(const DATensor *tensor);

  std::string name_;
  DAContext *context_{nullptr};
  DAGraph *graph_{nullptr};
  std::vector<DATensor *> parameters_;
#ifdef DUMP
  std::unordered_map<const DATensor *, ssize_t> paraNumMap_;
  std::unordered_map<const DATensor *, ssize_t> nodeNumMap_;
#endif
};
} // namespace runtime

#endif // __RUNTIME_EXECUTOR_H__