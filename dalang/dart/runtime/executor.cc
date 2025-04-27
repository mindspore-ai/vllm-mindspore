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

namespace da {
namespace runtime {
using namespace tensor;

GraphExecutor::GraphExecutor() : context_{tensor::NewDAContext()} {
  CHECK_IF_NULL(context_);
}

GraphExecutor::~GraphExecutor() {
  CHECK_IF_NULL(context_);
  FreeDAContext(context_);
  context_ = nullptr;
}

// Start building graph.
void GraphExecutor::BeginGraph(const std::string &name) {
  LOG_OUT << "Begin graph building";
  CHECK_IF_FAIL(graph_ == nullptr);
  CHECK_IF_NULL(context_);
  graph_ = tensor::NewDAGraph(context_);
  name_ = name;
}

// Finish building graph.
void GraphExecutor::EndGraph() {
  LOG_OUT << "End graph building";
  CHECK_IF_NULL(graph_);
}

// Add a parameter for graph.
void GraphExecutor::AddParameter(DATensor *param) {
  LOG_OUT << "Add parameter: " << param << " for graph: " << graph_;
  tensor::AddParameter(graph_, param);
}

// Add parameters for graph.
void GraphExecutor::AddParameters(const std::vector<DATensor *> &params) {
  LOG_OUT << "Add parameters[" << params.size() << "] for graph: " << graph_;
  for (const auto &param : params) {
    tensor::AddParameter(graph_, param);
  }
}

// Add a const tensor.
DATensor *GraphExecutor::AddTensor(Type type, size_t dim,
                                   const ShapeArray &shape, void *data) {
  LOG_OUT << "Add const tensor";
  CHECK_IF_NULL(context_);
  auto *tensor = tensor::NewDATensor(context_, type, dim, shape, data);
  CHECK_IF_NULL(tensor);
  if (graph_ != nullptr) {
    tensor::AddTensor(graph_, tensor);
  }
  return tensor;
}

// Add operation result tensor.
DATensor *GraphExecutor::AddTensor(ops::Op op,
                                   const std::vector<DATensor *> &inputs) {
  LOG_OUT << "Add tensor";
  auto tensorSize = inputs.size();
  LOG_OUT << "tensorSize: " << tensorSize;
  CHECK_IF_NULL(context_);
  auto *tensor = tensor::NewDATensor(context_);
  CHECK_IF_NULL(tensor);
  tensor->op = op;
  for (size_t i = 0; i < inputs.size(); ++i) {
    tensor->input[i] = inputs[i];
    ++tensor->inputSize;
  }
  CHECK_IF_NULL(graph_);
  tensor::AddTensor(graph_, tensor);
  return tensor;
}

void GraphExecutor::RunTensor(const DATensor *tensor) {
  LOG_OUT << "Run tensor, ops." << ops::ToStr(tensor->op)
          << ", tensor: " << tensor;
  // Call (tensor->op, tensor->node)
}

// Run the built graph.
void GraphExecutor::RunGraph() {
  LOG_OUT << "Run graph";
  CHECK_IF_NULL(context_);
  CHECK_IF_NULL(graph_);

  // Run all tensor nodes.
  DATensor *tensorNode{nullptr};
  for (size_t i = 0; i < graph_->nodeSize; ++i) {
    tensorNode = graph_->node[i];
    RunTensor(tensorNode);
  }
}

#undef DEBUG_DUMP
#ifdef DUMP
// Run the built graph.
void GraphExecutor::DumpGraph() {
  LOG_OUT << "Run graph";
  CHECK_IF_NULL(context_);
  CHECK_IF_NULL(graph_);

  constexpr auto paramPrefix = "param_";
  std::cout << "graph{" << name_ << "}(";
  for (size_t i = 0; i < graph_->paramSize; ++i) {
    auto para = graph_->param[i];
    paraNumMap_.emplace(para, i);
    std::cout << paramPrefix << i;
#ifdef DEBUG_DUMP
    std::cout << "(" << para << ")";
#endif
    if (i < graph_->paramSize - 1) {
      std::cout << ", ";
    }
  }
  std::cout << ") {" << std::endl;

  for (size_t i = 0; i < graph_->nodeSize; ++i) {
    nodeNumMap_.emplace(graph_->node[i], i);
  }

  // Run all tensor nodes.
  DATensor *tensorNode{nullptr};
  for (size_t i = 0; i < graph_->nodeSize; ++i) {
    tensorNode = graph_->node[i];
    size_t inputSize = tensorNode->inputSize;
    std::stringstream ss;
    for (size_t i = 0; i < inputSize; ++i) {
      auto input = tensorNode->input[i];
      // Find node number firstly.
      auto nodeIt = nodeNumMap_.find(input);
      if (nodeIt != nodeNumMap_.cend()) {
        ss << "%" << nodeIt->second;
      } else {
        // Find parameter number.
        auto paraIt = paraNumMap_.find(input);
        if (paraIt != paraNumMap_.cend()) {
          ss << paramPrefix << paraIt->second;
        } else {
          ss << "<ERR>";
        }
      }
#ifdef DEBUG_DUMP
      ss << "(" << input << ")";
#endif
      if (i != inputSize - 1) {
        ss << ", ";
      }
    }

    if (nodeNumMap_.count(tensorNode) == 0) {
      LOG_ERROR << "Failed to find tensor number for " << tensorNode;
      exit(EXIT_FAILURE);
    }
    std::cout << "  %" << nodeNumMap_[tensorNode];
#ifdef DEBUG_DUMP
    std::cout << "(" << tensorNode << ")";
#endif
    std::cout << " = ops." << ops::ToStr(tensorNode->op) << "(" << ss.str()
              << ")" << std::endl;
  }

  std::cout << "  return %" << nodeNumMap_[tensorNode] << std::endl;
  std::cout << "}" << std::endl;
}
#endif
} // namespace runtime
} // namespace da
