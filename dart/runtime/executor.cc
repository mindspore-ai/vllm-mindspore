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

#include "runtime/executor.h"

#include <sstream>

namespace runtime {

void GraphExecutor::BeginGraph() {
  LOG_OUT << "begin graph";
  CHECK_FAIL(graph_ == nullptr);
  CHECK_NULL(context_);
  graph_ = NewDAGraph(context_);
}

void GraphExecutor::EndGraph() {
  LOG_OUT << "end graph";
  CHECK_NULL(graph_);
}

// Add a const tensor.
DATensor *GraphExecutor::AddTensor(Type type, size_t dim,
                                   size_t shape[DA_TENSOR_MAX_DIM],
                                   void *data) {
  LOG_OUT << "add const tensor";
  CHECK_NULL(context_);
  auto *tensor = NewDATensor(context_, type, dim, shape, data);
  CHECK_NULL(tensor);
  if (graph_ != nullptr) {
    graph_->node[graph_->num] = tensor;
    ++graph_->num;
  }
  return tensor;
}

// Add operation result tensor.
DATensor *GraphExecutor::AddTensor(const std::vector<DATensor *> &inputs) {
  LOG_OUT << "add tensor";
  CHECK_NULL(context_);
  auto *tensor = NewDATensor(context_);
  CHECK_NULL(tensor);
  for (const auto &input : inputs) {
    tensor->input[tensor->inputNum] = input;
    ++tensor->inputNum;
  }
  if (graph_ != nullptr) {
    graph_->node[graph_->num] = tensor;
    ++graph_->num;
  }
  return tensor;
}

void GraphExecutor::RunTensor(const DATensor *tensor) {
  // Call (tensor->op, tensor->node)
#ifdef DEBUG
  std::stringstream ss;
  for (size_t i = 0; i < tensor->inputNum; ++i) {
    ss << "%" << tensorNumMap_[tensor->input[i]];
    if (i != tensor->inputNum - 1) {
      ss << ", ";
    }
  }
  std::cout << "  %" << tensorNumMap_[tensor] << " = ops."
            << ops::ToStr(tensor->op) << "(" << ss.str() << ")" << std::endl;
#endif
}

void GraphExecutor::Run() {
  CHECK_NULL(context_);
  CHECK_NULL(graph_);
#ifdef DEBUG
  for (size_t i = 0; i < graph_->num; ++i) {
    tensorNumMap_.emplace(graph_->node[i], i);
  }
  std::cout << "graph(" << graph_->num << ") {" << std::endl;
#endif
  // Run all tensor nodes.
  for (size_t i = 0; i < graph_->num; ++i) {
    auto tensorNode = graph_->node[i];
    RunTensor(tensorNode);
  }

#ifdef DEBUG
  std::cout << "}" << std::endl;
#endif
}
} // namespace runtime
