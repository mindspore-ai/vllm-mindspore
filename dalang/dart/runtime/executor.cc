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

#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <mutex>
#include <queue>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <vector>

#include "runtime/executor.h"

namespace da {
namespace runtime {
using namespace tensor;

const std::vector<std::string> GetEnvKernelLibPaths() {
  std::vector<std::string> kernelLibPaths{};
  constexpr char kKernelLibPathsEnvName[] = "DART_KERNEL_LIB_PATH";
  const char *pathsCStr = std::getenv(kKernelLibPathsEnvName);
  if (pathsCStr == nullptr) {
    return kernelLibPaths;
  }

  size_t pathLen = 0;
  while (pathsCStr[pathLen] != '\0') {
    if (pathsCStr[pathLen] == ',') {
      (void)kernelLibPaths.emplace_back(std::string(pathsCStr, pathLen));
      pathsCStr += pathLen + 1;
      pathLen = 0;
    } else {
      ++pathLen;
    }
  }
  (void)kernelLibPaths.emplace_back(std::string(pathsCStr, pathLen));
  return kernelLibPaths;
}

const std::string GetEnvKernelLibName() {
  constexpr char kDefaultKernelLibName[] = "Mindspore";
  constexpr char kKernelLibEnvName[] = "DART_KERNEL_LIB_NAME";
  const char *name = std::getenv(kKernelLibEnvName);
  if (name == nullptr) {
    name = kDefaultKernelLibName;
  }
  return name;
}

size_t GetEnvThreadPoolSize() {
  const size_t kDefaultThreadPoolSize = 1;
  constexpr char kThreadPoolSizeEnvName[] = "DART_THREAD_POOL_SIZE";
  const char *poolSizeStr = std::getenv(kThreadPoolSizeEnvName);
  if (poolSizeStr == nullptr) {
    return kDefaultThreadPoolSize;
  }
  return std::stoul(poolSizeStr);
}

GraphExecutor::GraphExecutor() : context_{tensor::NewDAContext()} {
  CHECK_IF_NULL(context_);

  memPool_ = new MemoryPool();
  CHECK_IF_NULL(memPool_);

  for (auto &&path : GetEnvKernelLibPaths()) {
    KernelLibRegistry::Instance().Load(path);
  }
}

GraphExecutor::~GraphExecutor() {
  CHECK_IF_NULL(context_);
  FreeDAContext(context_);
  context_ = nullptr;

  CHECK_IF_NULL(memPool_);
  delete memPool_;
  memPool_ = nullptr;
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

// Add tensor for graph
void GraphExecutor::AddTensor(DATensor *tensor) {
  LOG_OUT << "Add const tensor";
  CHECK_IF_NULL(context_);
  CHECK_IF_NULL(tensor);
  CHECK_IF_NULL(graph_);
  tensor::AddTensor(graph_, tensor);
}

// Add tensor list to tensor.
void GraphExecutor::CastToTensorList(DATensor *tensor, size_t len) {
  LOG_OUT << "Add tensor list";
  CHECK_IF_NULL(context_);
  CHECK_IF_NULL(tensor);
  tensor->type = tensor::Type_Tensor;
  tensor->dim = 1;
  tensor->shape[0] = len;
  auto **tensorList = tensor::NewDATensorList(context_, len);
  tensor->data = static_cast<void *>(tensorList);
}

// Add operation result tensor.
DATensor *GraphExecutor::AddTensor(ops::Op op,
                                   const std::vector<DATensor *> &inputs) {
  LOG_OUT << "Add tensor";
  auto tensorSize = inputs.size();
  LOG_OUT << "tensor input size: " << tensorSize;
  CHECK_IF_FAIL(tensorSize <= DA_TENSOR_MAX_INPUT);
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

// Run DATensor node
void GraphExecutor::RunTensor(DATensor *node) {
  if (node->op == ops::Op_End) {
    return;
  }

  if (node->op == ops::Op_make_tuple) {
    CHECK_IF_FAIL(node->type == Type_Tensor);
    CHECK_IF_FAIL(node->shape[0] == node->inputSize);
    auto **tensorList = static_cast<DATensor **>(node->data);
    CHECK_IF_NULL(tensorList);
    for (size_t i = 0; i < node->inputSize; ++i) {
      std::unique_lock<std::mutex> lock(outputsMutex_);
      tensorList[i]->data = node->input[i]->data;
      CloneDATensorShape(tensorList[i], node->input[i]);
    }
    return;
  }

  if (node->op == ops::Op_tuple_getitem) {
    CHECK_IF_FAIL(node->input[kFirstInput]->type == Type_Tensor);
    auto index =
        static_cast<size_t>(GetValue<int64_t>(node->input[kSecondInput]));
    std::unique_lock<std::mutex> lock(outputsMutex_);
    LOG_OUT << "Run Op_tuple_getitem, tensor: " << node
            << ", input_tensor: " << node->input[kFirstInput]
            << ", index_tensor: " << node->input[kSecondInput]
            << ", index: " << index;
    CHECK_IF_FAIL(index < node->input[kFirstInput]->shape[0]);
    auto **inputTensorList =
        static_cast<DATensor **>(node->input[kFirstInput]->data);
    CHECK_IF_NULL(inputTensorList);
    node->data = inputTensorList[index]->data;
    CloneDATensorShape(node, inputTensorList[index]);
    return;
  }

  if (IsDATensorOutputFromInput(node)) {
    auto inputIndex = GetDATensorOuputFromInputIndex(node);
    std::unique_lock<std::mutex> lock(outputsMutex_);
    node->data = node->input[inputIndex]->data;
    CloneDATensorShape(node, node->input[inputIndex]);
    return;
  }

  // Get real inputs of the node.
  GetNodeRealInputs(node);

#ifndef SKIP_RUN_TENSOR
  auto kernelLib = KernelLibRegistry::Instance().Get(GetEnvKernelLibName());
  CHECK_IF_NULL(kernelLib);
  CHECK_IF_FAIL(kernelLib->RunTensor(node, memPool_));
#endif
}

// Run the built graph.
void GraphExecutor::RunGraph() {
  LOG_OUT << "Run graph";
  CHECK_IF_NULL(context_);
  CHECK_IF_NULL(graph_);

  memPool_->Reset();

  // record ref counts
  TensorDataRecycler recycler(memPool_);
  for (size_t i = 0; i < graph_->nodeSize; ++i) {
    recycler.ForwardRecordInputsRefCounts(graph_->node[i]);
  }
  recycler.PrintRefCountInfo();

#ifndef SERIAL
  for (size_t i = 0; i < graph_->nodeSize; ++i) {
    LOG_OUT << "Run tensor, ops." << ops::ToStr(graph_->node[i]->op)
            << ", DATensor: " << graph_->node[i] << ", index: " << i;
    RunTensor(graph_->node[i]);
#ifndef SKIP_RUN_TENSOR
    recycler.DecreaseInputsRefCounts(graph_->node[i]);
  }
  recycler.CheckRefCountInfo();
#else
  }
#endif
#else
  std::unordered_map<DATensor *, size_t> waitingCount;
  std::unordered_map<DATensor *, std::vector<DATensor *>> nextNodes;
  std::queue<DATensor *> readyQueue;

  // Initialize execution DAG.
  for (size_t i = 0; i < graph_->nodeSize; ++i) {
    auto node = graph_->node[i];
    for (size_t j = 0; j < node->inputSize; ++j) {
      (void)nextNodes[node->input[j]].emplace_back(node);
    }
  }
  // Initialize ready queue with parameters nodes and nodes with no dependencies.
  for (size_t i = 0; i < graph_->paramSize; ++i) {
    (void)readyQueue.emplace(graph_->param[i]);
  }
  for (size_t i = 0; i < graph_->nodeSize; ++i) {
    auto node = graph_->node[i];
    waitingCount[node] = node->inputSize;
    if (waitingCount[node] == 0) {
      (void)readyQueue.emplace(node);
    }
  }

  size_t runningCount = 0;
  std::mutex mutex;
  std::condition_variable cv;
  auto worker = [&]() {
    while (true) {
      DATensor *node = nullptr;
      {
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait(lock, [&]() { return runningCount == 0 || !readyQueue.empty(); });
        if (runningCount == 0 && readyQueue.empty()) {
          return;
        }
        node = readyQueue.front();
        readyQueue.pop();
        runningCount++;
      }

      RunTensor(node);

      {
        std::lock_guard<std::mutex> lock(mutex);
        for (auto next : nextNodes[node]) {
          if (--waitingCount[next] == 0) {
            (void)readyQueue.emplace(next);
          }
        }
        runningCount--;
        cv.notify_all();
      }
    }
  };

  std::vector<std::thread> thread_pool;
  for (size_t i = 0; i < GetEnvThreadPoolSize(); ++i) {
    (void)thread_pool.emplace_back(worker);
  }
  for (auto &thread : thread_pool) {
    thread.join();
  }
#endif
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
