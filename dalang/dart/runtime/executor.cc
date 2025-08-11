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
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "runtime/executor.h"

namespace da {
namespace runtime {
using namespace tensor;
namespace {
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

#ifndef SKIP_RUN_TENSOR
const std::string GetEnvKernelLibName() {
  constexpr char kDefaultKernelLibName[] = "Mindspore";
  constexpr char kKernelLibEnvName[] = "DART_KERNEL_LIB_NAME";
  const char *name = std::getenv(kKernelLibEnvName);
  if (name == nullptr) {
    name = kDefaultKernelLibName;
  }
  return name;
}
#endif

size_t GetEnvThreadPoolSize() {
  const size_t kDefaultThreadPoolSize = 1;
  constexpr char kThreadPoolSizeEnvName[] = "DART_THREAD_POOL_SIZE";
  const char *poolSizeStr = std::getenv(kThreadPoolSizeEnvName);
  if (poolSizeStr == nullptr) {
    return kDefaultThreadPoolSize;
  }
  return std::stoul(poolSizeStr);
}

void ProcessMakeTuple(DATensor *makeTupleNode) {
  CHECK_IF_NULL(makeTupleNode);
  CHECK_IF_FAIL(makeTupleNode->type == Type_Tensor);
  CHECK_IF_FAIL(makeTupleNode->shape[0] == makeTupleNode->inputSize);
  auto **tensorList = static_cast<DATensor **>(makeTupleNode->data);
  CHECK_IF_NULL(tensorList);
  for (size_t i = 0; i < makeTupleNode->inputSize; ++i) {
    tensorList[i]->data = makeTupleNode->input[i]->data;
    CloneDATensorShape(tensorList[i], makeTupleNode->input[i]);
  }
}

void ProcessTupleGetItem(DATensor *tupleGetItemNode) {
  CHECK_IF_NULL(tupleGetItemNode)
  CHECK_IF_FAIL(tupleGetItemNode->input[kFirstInput]->type == Type_Tensor);
  auto index = static_cast<size_t>(
      GetValue<int64_t>(tupleGetItemNode->input[kSecondInput]));
  LOG_OUT << "Run Op_tuple_getitem, tensor: " << tupleGetItemNode
          << ", input_tensor: " << tupleGetItemNode->input[kFirstInput]
          << ", index_tensor: " << tupleGetItemNode->input[kSecondInput]
          << ", index: " << index;
  CHECK_IF_FAIL(index < tupleGetItemNode->input[kFirstInput]->shape[0]);
  auto **inputTensorList =
      static_cast<DATensor **>(tupleGetItemNode->input[kFirstInput]->data);
  CHECK_IF_NULL(inputTensorList);
  tupleGetItemNode->data = inputTensorList[index]->data;
  CloneDATensorShape(tupleGetItemNode, inputTensorList[index]);
}

void ProcessOutputFromInput(DATensor *outputFromInputNode) {
  CHECK_IF_NULL(outputFromInputNode);
  auto inputIndex = GetDATensorOuputFromInputIndex(outputFromInputNode);
  outputFromInputNode->data = outputFromInputNode->input[inputIndex]->data;
  CloneDATensorShape(outputFromInputNode,
                     outputFromInputNode->input[inputIndex]);
}
} // namespace

GraphExecutor::GraphExecutor() : context_{tensor::NewDAContext()} {
  CHECK_IF_NULL(context_);

  recycler_ = new TensorDataRecycler();
  CHECK_IF_NULL(recycler_);

  for (auto &&path : GetEnvKernelLibPaths()) {
    KernelLibRegistry::Instance().Load(path);
  }
}

GraphExecutor::~GraphExecutor() {
  CHECK_IF_NULL(context_);
  FreeDAContext(context_);
  context_ = nullptr;

  CHECK_IF_NULL(recycler_);
  delete recycler_;
  recycler_ = nullptr;
  for (auto &kernelPair : kernels_) {
    CHECK_IF_NULL(kernelPair.second);
    delete kernelPair.second;
  }
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

// Finish building graph.
void GraphExecutor::OptGraph() {
  LOG_OUT << "Opt graph";
  CHECK_IF_NULL(graph_);
  pass::TensorCreator tensorCreator =
      std::bind((DATensor * (GraphExecutor::*)(ops::Op, DATensor **, size_t)) &
                    GraphExecutor::AddTensor,
                this, std::placeholders::_1, std::placeholders::_2,
                std::placeholders::_3);
  pass::PassManager::Instance().Run(graph_, tensorCreator);
}

// Build DAKernels for graph.
void GraphExecutor::BuildKernels() {
  CHECK_IF_NULL(graph_);
#ifndef SKIP_RUN_TENSOR
  auto kernelLib = KernelLibRegistry::Instance().Get(GetEnvKernelLibName());
  CHECK_IF_NULL(kernelLib);
  for (size_t i = 0; i < graph_->nodeSize; ++i) {
    auto node = graph_->node[i];
    CHECK_IF_NULL(node);
    if (IsSkipBuildDAKernel(node)) {
      continue;
    }
    // Get real inputs of the node.
    GetNodeRealInputs(node);
    auto kernel = kernelLib->CreateKernel(node);
    CHECK_IF_NULL(kernel);
    kernel->Init();
    kernels_[node] = kernel;
  }
#endif
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

// Add operation result tensor.
DATensor *GraphExecutor::AddTensor(ops::Op op, DATensor **start, size_t size) {
  LOG_OUT << "Add tensor";
  const auto tensorListSize = size;
  LOG_OUT << "tensor input size: " << tensorListSize;
  CHECK_IF_FAIL(tensorListSize <= DA_TENSOR_MAX_INPUT);
  CHECK_IF_NULL(context_);
  auto *tensor = tensor::NewDATensor(context_);
  CHECK_IF_NULL(tensor);
  tensor->op = op;
  for (size_t i = 0; i < tensorListSize; ++i) {
    tensor->input[i] = start[i];
    ++tensor->inputSize;
  }
  CHECK_IF_NULL(graph_);
  tensor::AddTensor(graph_, tensor);
  return tensor;
}

// Add operation result tensor.
DATensor *GraphExecutor::AddTensor(ops::Op op,
                                   const std::vector<DATensor *> &inputs) {
  LOG_OUT << "Add tensor";
  LOG_OUT << "tensor input size: " << inputs.size();
  CHECK_IF_FAIL(inputs.size() <= DA_TENSOR_MAX_INPUT);
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

// Add return node.
DATensor *GraphExecutor::AddReturn() {
  LOG_OUT << "Add return";
  CHECK_IF_NULL(context_);
  auto *tensor = tensor::NewDATensor(context_);
  CHECK_IF_NULL(tensor);
  tensor->op = ops::Op_return;
  tensor->inputSize = 1;
  CHECK_IF_NULL(graph_);
  CHECK_IF_FAIL(graph_->nodeSize > 0);
  tensor->input[0] = graph_->node[graph_->nodeSize - 1];
  tensor::AddTensor(graph_, tensor);
  return tensor;
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

// Run DATensor node
void GraphExecutor::RunTensor(DATensor *node) {
  if (node->op == ops::Op_End) {
    return;
  }

  if (node->op == ops::Op_make_tuple) {
    ProcessMakeTuple(node);
    return;
  }

  if (node->op == ops::Op_tuple_getitem) {
    ProcessTupleGetItem(node);
    return;
  }

  if (IsDATensorOutputFromInput(node)) {
    ProcessOutputFromInput(node);
    return;
  }

#ifndef SKIP_RUN_TENSOR
  CHECK_IF_FAIL(kernels_.find(node) != kernels_.end());
  kernels_[node]->RunKernel(isDynamic_);
  if (node->op != ops::Op_return) {
    // keep outputs memory until consumed.
    recycler_->FreeUnusedNodes(node);
  }
#endif
}

// Free the memory of graph outputs
void GraphExecutor::FreeGraphOutputs() {
  CHECK_IF_NULL(graph_);
  CHECK_IF_NULL(recycler_);
  auto returnNode = graph_->node[graph_->nodeSize-1];
  CHECK_IF_FAIL(returnNode->op == ops::Op_return);
  recycler_->FreeUnusedNodes(returnNode);
  recycler_->PrintRunningRefCounts();
}

// Record tensor refCount
void GraphExecutor::RecordTensorRefCount() {
  CHECK_IF_NULL(recycler_);
  CHECK_IF_NULL(graph_);
  CHECK_IF_NULL(context_);

  for (size_t i = 0; i < graph_->nodeSize; ++i) {
    recycler_->ForwardRecordInputsRefCounts(graph_->node[i]);
  }
}

// Run the built graph.
void GraphExecutor::RunGraph(bool isDynamic) {
  LOG_OUT << "Run graph, isDynamic: " << isDynamic;
  CHECK_IF_NULL(context_);
  CHECK_IF_NULL(graph_);
  CHECK_IF_NULL(recycler_);

  isDynamic_ = isDynamic;
  recycler_->ResetRunningRefCounts();

#ifdef SERIAL
  for (size_t i = 0; i < graph_->nodeSize; ++i) {
    LOG_OUT << "Run tensor, ops." << ops::ToStr(graph_->node[i]->op)
            << ", DATensor: " << graph_->node[i] << ", index: " << i;
    RunTensor(graph_->node[i]);
  }
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
  // Initialize ready queue with parameters nodes and nodes with no
  // dependencies.
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
        cv.wait(lock,
                [&]() { return runningCount == 0 || !readyQueue.empty(); });
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
