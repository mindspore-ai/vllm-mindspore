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

#include "runtime/executor/executor.h"

#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <mutex>
#include <queue>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <vector>

#include "ops/kernel_lib.h"
#include "runtime/builder/builder.h"
#include "runtime/builder/pipeline/pipeline_builder.h"

namespace mrt {
namespace runtime {
using BuilderCreationFunc = std::function<std::unique_ptr<Builder>(const ir::GraphPtr &)>;
static std::vector<BuilderCreationFunc> builderCreators = {
  [](const ir::GraphPtr &graph) { return std::make_unique<Builder>(graph); },
  [](const ir::GraphPtr &graph) { return std::make_unique<PipelineBuilder>(graph); }};

namespace {
ExecutionMode GetExecutionMode() {
  static const char enablePipelineEnv[] = "MRT_ENABLE_PIPELINE";
  static const char *enablePipelineCStr = std::getenv(enablePipelineEnv);
  static const bool enablePipeline = (enablePipelineCStr != nullptr) && (std::string_view(enablePipelineCStr) == "on");
  ExecutionMode executionMode = Base;
  if (enablePipeline) {
    executionMode = Pipeline;
  }
  return executionMode;
}

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

void ProcessMakeTuple(ir::NodePtr node) {
  CHECK_IF_NULL(node);
  std::vector<ir::ValuePtr> elements;
  for (auto &input : node->inputs) {
    (void)elements.emplace_back(input->output);
  }
  node->output = ir::MakeIntrusive<ir::Value>(ir::MakeIntrusive<ir::Tuple>(std::move(elements)));
}

void ProcessTupleGetItem(ir::NodePtr node) {
  CHECK_IF_NULL(node);
  auto index = node->inputs[kSecondInput]->output->ToInt();
  auto tuple = node->inputs[kFirstInput]->output->ToTuple();
  CHECK_IF_FAIL(static_cast<size_t>(index) < tuple->Size());
  node->output = (*tuple)[index];
}
}  // namespace

GraphExecutor::GraphExecutor() {
  recycler_ = new TensorDataRecycler();
  CHECK_IF_NULL(recycler_);

  for (auto &&path : GetEnvKernelLibPaths()) {
    ops::KernelLibRegistry::Instance().Load(path);
  }
}

GraphExecutor::~GraphExecutor() {
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
  graph_ = ir::MakeIntrusive<ir::Graph>();
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
    std::bind((ir::NodePtr (GraphExecutor::*)(ops::Op, const std::vector<ir::NodePtr> &,
                                              const ir::ValuePtr &))&GraphExecutor::AddOpNode,
              this, std::placeholders::_1, std::placeholders::_2, nullptr);
  pass::PassManager::Instance().Run(graph_, tensorCreator);
}

// Build DAKernels for graph.
void GraphExecutor::BuildKernels() {
  CHECK_IF_NULL(graph_);
  auto kernelLib = ops::KernelLibRegistry::Instance().Get(GetEnvKernelLibName());
  CHECK_IF_NULL(kernelLib);
  for (auto &node : graph_->nodes) {
    CHECK_IF_NULL(node);
    if (IsSkipBuildDAKernel(node)) {
      continue;
    }
    auto kernel = kernelLib->CreateKernel(node);
    CHECK_IF_NULL(kernel);
    kernel->Init();
    kernels_[node] = kernel;
  }
}

// Add a node as parameter for graph.
void GraphExecutor::AddParameter(ir::NodePtr param) {
  LOG_OUT << "Add parameter: " << param;
  CHECK_IF_NULL(param);
  CHECK_IF_FAIL(param->op == ops::Op_End);
  (void)parameters_.emplace_back(param);
}

// Add a value node.
ir::NodePtr GraphExecutor::AddValueNode(const ir::ValuePtr &value) {
  LOG_OUT << "Add value node: " << value;
  auto node = ir::MakeIntrusive<ir::Node>();
  node->op = ops::Op_End;
  node->output = value == nullptr ? ir::MakeIntrusive<ir::Value>() : value;
  if (graph_ != nullptr) {
    (void)graph_->nodes.emplace_back(node);
  }
  return node;
}

// Add an operation node.
ir::NodePtr GraphExecutor::AddOpNode(ops::Op op, const std::vector<ir::NodePtr> &inputs, const ir::ValuePtr &output) {
  LOG_OUT << "Add operation node";
  LOG_OUT << "operation input size: " << inputs.size();
  auto node = ir::MakeIntrusive<ir::Node>();
  CHECK_IF_NULL(node);
  node->op = op;
  node->inputs = inputs;
  node->output = output == nullptr ? ir::MakeIntrusive<ir::Value>() : output;
  CHECK_IF_NULL(graph_);
  (void)graph_->nodes.emplace_back(node);
  return node;
}

// Add return node.
ir::NodePtr GraphExecutor::AddReturn() {
  LOG_OUT << "Add return node from last node";
  CHECK_IF_NULL(graph_);
  auto lastNode = graph_->nodes[graph_->nodes.size() - 1];
  auto node = ir::MakeIntrusive<ir::Node>();
  CHECK_IF_NULL(node);
  node->op = ops::Op_return;
  node->output = lastNode->output;
  (void)node->inputs.emplace_back(lastNode);
  (void)graph_->nodes.emplace_back(node);
  return node;
}

// Run a single node
void GraphExecutor::RunNode(ir::NodePtr node) {
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

  if (auto it = opsOutputFromInputIndex.find(node->op); it != opsOutputFromInputIndex.end()) {
    node->output = node->inputs[it->second]->output;
    return;
  }

  auto iter = kernels_.find(node);
  if (iter == kernels_.end()) {
    LOG_ERROR << "kernel not found: " << node;
    exit(EXIT_FAILURE);
  }
  auto kernel = iter->second;

  if (isDynamic_) {
    kernel->InferShape();
    kernel->Resize();
  } else if (IsDAKernelNeedForceResize(node)) {
    kernel->Resize();
  }

  if (auto it = opsOutputValueFromInputIndex.find(node->op); it != opsOutputValueFromInputIndex.end()) {
    LOG_OUT << "Skip launch kernel for node" << node;
    auto outputTensor = node->output->ToTensor();
    auto inputStorage = node->inputs[it->second]->output->ToTensor()->GetStorage();
    node->output = ir::MakeIntrusive<ir::Value>(
      ir::MakeIntrusive<ir::Tensor>(inputStorage, outputTensor->Shape(), outputTensor->Dtype()));
  } else {
    kernel->Launch();
  }

  if (node->op != ops::Op_return) {
    // keep outputs memory until consumed.
    recycler_->FreeUnusedNodes(node);
  }
}

// Free the memory of graph outputs
void GraphExecutor::FreeGraphOutputs() {
  CHECK_IF_NULL(graph_);
  CHECK_IF_NULL(recycler_);
  auto returnNode = graph_->nodes[graph_->nodes.size() - 1];
  CHECK_IF_FAIL(returnNode->op == ops::Op_return);
  recycler_->FreeUnusedNodes(returnNode);
  recycler_->PrintRunningRefCounts();
}

// Record tensor refCount
void GraphExecutor::RecordTensorRefCount() {
  CHECK_IF_NULL(recycler_);
  CHECK_IF_NULL(graph_);

  for (auto &node : graph_->nodes) {
    recycler_->ForwardRecordInputsRefCounts(node);
  }
}

// Run the built graph.
void GraphExecutor::RunGraph(bool isDynamic) {
  CHECK_IF_NULL(executor_);
  LOG_OUT << "Run graph, isDynamic: " << isDynamic;
  executor_->Run(isDynamic);
}

#ifdef DUMP
// Run the built graph.
void GraphExecutor::DumpGraph() {
  LOG_OUT << "Run graph";
  CHECK_IF_NULL(graph_);

  constexpr auto paramPrefix = "param_";
  std::cout << "graph{" << name_ << "}(";
  for (size_t i = 0; i < parameters_.size(); ++i) {
    auto para = parameters_[i];
    (void)paraNumMap_.emplace(para, i);
    std::cout << paramPrefix << i;
    if (i < parameters_.size() - 1) {
      std::cout << ", ";
    }
  }
  std::cout << ") {" << std::endl;

  for (size_t i = 0; i < graph_->nodes.size(); ++i) {
    (void)nodeNumMap_.emplace(graph_->nodes[i], i);
  }

  // Run all tensor nodes.
  ir::NodePtr tensorNode{nullptr};
  for (size_t i = 0; i < graph_->nodes.size(); ++i) {
    tensorNode = graph_->nodes[i];
    size_t inputSize = tensorNode->inputs.size();
    std::stringstream ss;
    for (size_t j = 0; j < inputSize; ++j) {
      auto input = tensorNode->inputs[j];
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
      if (j != inputSize - 1) {
        ss << ", ";
      }
    }

    if (nodeNumMap_.count(tensorNode) == 0) {
      LOG_ERROR << "Failed to find tensor number for " << tensorNode;
      exit(EXIT_FAILURE);
    }
    std::cout << "  %" << nodeNumMap_[tensorNode];
    std::cout << " = ops." << ops::ToStr(tensorNode->op) << "(" << ss.str() << ")";
    std::cout << std::setw(10) << "// " << tensorNode->output << std::endl;
  }

  std::cout << "  return %" << nodeNumMap_[tensorNode] << std::endl;
  std::cout << "}" << std::endl;
}
#endif

void GraphExecutor::BuildExecutor() {
  CHECK_IF_FAIL(nullptr == builder_);
  CHECK_IF_FAIL(nullptr == executor_);
  ExecutionMode executionMode = GetExecutionMode();
  CHECK_IF_FAIL(static_cast<size_t>(executionMode) < builderCreators.size());
  builder_ = builderCreators.at(static_cast<size_t>(executionMode))(graph_);
  CHECK_IF_NULL(builder_);
  executor_ = builder_->BuildExecutor();
  CHECK_IF_NULL(executor_);
}

void Executor::Run(bool isDynamic) {
  OpRunner *opRunners = opRunners_->data();
  size_t opNum = opRunners_->size();
  for (size_t i = 0; i < opNum; i++) {
    OpRunner &opRunner = opRunners[i];
    if (auto errNo = opRunner.InferShape() != ops::SUCCESS) {
      LOG_EXCEPTION << "Infer shape failed for operator " << ops::ToStr(opRunner.GetNode()->op) << "Errno: " << errNo;
    }
    if (auto errNo = opRunner.CalcWorkspace() != ops::SUCCESS) {
      LOG_EXCEPTION << "CalcWorkspace shape failed for operator " << ops::ToStr(opRunner.GetNode()->op)
                    << "Errno: " << errNo;
    }
    if (auto errNo = opRunner.Launch() != ops::SUCCESS) {
      LOG_EXCEPTION << "Launch shape failed for operator " << ops::ToStr(opRunner.GetNode()->op) << "Errno: " << errNo;
    }
  }

  for (auto &deviceItem : deviceContexts_) {
    const auto &res_manager = deviceItem.second->deviceResManager_;
    CHECK_IF_NULL(res_manager);
    res_manager->SyncAllStreams();
  }
}
}  // namespace runtime
}  // namespace mrt
