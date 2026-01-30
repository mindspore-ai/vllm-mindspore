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

#include "runtime/builder/builder.h"
#include <algorithm>
#include <unordered_set>
#include <memory>
#include <utility>
#include "runtime/executor/executor.h"
#include "ops/op_register.h"
#include "hardware/hardware_abstract/device_context_manager.h"
#include "hardware/hardware_abstract/collective/collective_manager.h"

namespace mrt {
namespace runtime {

namespace {
/**
 * @brief Get the device type of an operation node.
 *  The function follows these rules:
 * 1. If the output is a single tensor, returns the device of that tensor.
 * 2. If the output is None:
 *    - If any input element is a tensor, returns the device of the first tensor.
 *    - If no input element is a tensor, defaults to CPU device.
 * 3. If the output is a tuple:
 *    - If all elements are tensors, returns the device of the first tensor.
 *    - If any element is not a tensor, defaults to CPU device.
 * 4. For any other case or if checks fail, defaults to CPU device.
 * Note: Need to select operator type as 'device' for the copy operator.
 */
hardware::Device GetOpDeviceType(const ir::NodePtr &opNode) {
  const ir::ValuePtr &nodeOutput = opNode->output;
  CHECK_IF_NULL(nodeOutput);
  if (nodeOutput->IsTensor()) {
    auto &tensor = nodeOutput->ToTensor();
    CHECK_IF_NULL(tensor);
    return tensor->GetDevice();
  }

  if (nodeOutput->IsNone()) {
    auto &inputs = opNode->inputs;
    auto it = std::find_if(inputs.begin(), inputs.end(), [](const auto &node) { return node->output->IsTensor(); });
    if (it != inputs.end()) {
      return (*it)->output->ToTensor()->GetDevice();
    }
  }

  if (nodeOutput->IsTuple()) {
    auto &tuple = nodeOutput->ToTuple();
    CHECK_IF_NULL(tuple);

    // Empty tuple (e.g., from mrt.shape on scalar tensors) - default to CPU.
    if (tuple->Size() == 0) {
      return {hardware::DeviceType::CPU, 0};
    }

    bool allTensor = std::all_of(tuple->begin(), tuple->end(), [](const ir::ValuePtr &elem) {
      CHECK_IF_NULL(elem);
      return elem->IsTensor();
    });

    if (!allTensor) {
      // Mixed or non-tensor types in tuple - default to CPU.
      return {hardware::DeviceType::CPU, 0};
    } else {
      // All elements are tensors - use first tensor's device.
      return (*tuple->begin())->ToTensor()->GetDevice();
    }
  }

  // CPU for any other output type.
  return {hardware::DeviceType::CPU, 0};
}
}  // namespace

std::unique_ptr<Executor> Builder::BuildExecutor() {
  SetupOpRunners();
  return std::make_unique<Executor>(opRunners_, deviceContexts_);
}

void Builder::SetupOpRunners() {
  CreateOpRunners();
  UpdateRefNodeOutputValue();
  RecordStorageFreePoint();
}

void Builder::UpdateRefNodeOutputValue() {
  auto &opRunners = *opRunners_;
  for (auto &opRunner : opRunners) {
    opRunner.UpdateRefNodeOutputValue();
  }
}

void Builder::RecordStorageFreePoint() {
  if (graph_ == nullptr) {
    return;
  }

  // Step 1.
  // First, create a map to record the first node that sees each storage (the storage's owner)
  std::unordered_map<ir::Storage *, ir::Node *> storageToOwner;  // key: storage pointer, value: owning node
  // Visit all parameters in the graph
  for (auto &param : graph_->parameters) {
    ir::VisitAllTensors(param->output, [&](const ir::TensorPtr &tensor) {
      auto storage = tensor->GetStorage().get();
      CHECK_IF_NULL(storage);
      (void)storageToOwner.insert({storage, param.get()});
    });
  }
  // Visit all nodes in the graph
  for (auto &node : graph_->nodes) {
    CHECK_IF_NULL(node);
    ir::VisitAllTensors(node->output, [&](const ir::TensorPtr &tensor) {
      auto storage = tensor->GetStorage().get();
      CHECK_IF_NULL(storage);
      (void)storageToOwner.insert({storage, node.get()});
    });
  }

  // Step 2.
  // The storages that can be safely freed once the last consumer node is done.
  std::unordered_map<ir::Node *, std::vector<ir::Storage *>> storagesToFree;
  // The storages that should be allocated by each node.
  std::unordered_map<ir::Node *, std::vector<ir::Storage *>> storagesToAlloc;
  std::unordered_set<ir::Storage *> recordedStorages;

  // The output of graph should not be freed by any node.
  auto &graphOutput = graph_->nodes.back()->output;
  ir::VisitAllTensors(graphOutput, [&](const ir::TensorPtr &tensor) {
    auto storage = tensor->GetStorage().get();
    CHECK_IF_NULL(storage);
    LOG_OUT << "Record graph output Storage: " << storage;
    (void)recordedStorages.insert(storage);
  });

  // Traverse in reverse execution order
  for (auto iter = graph_->nodes.rbegin(); iter != graph_->nodes.rend(); ++iter) {
    LOG_OUT << "Node: " << *iter;
    auto currentNode = iter->get();
    CHECK_IF_NULL(currentNode);
    if (IsSkipBuildDAKernel(currentNode)) {
      continue;
    }

    // Each op node is responsible for freeing the storage of its inputs.
    for (auto &inputNode : currentNode->inputs) {
      LOG_OUT << "Input: " << inputNode;
      ir::VisitAllTensors(inputNode->output, [&](const ir::TensorPtr &tensor) {
        auto storage = tensor->GetStorage().get();
        CHECK_IF_NULL(storage);
        // No longer check storage->CheckOwnsData()
        // First encounter, meaning current node is the last consumer
        // and is responsible for freeing the storage.
        if (recordedStorages.find(storage) == recordedStorages.end()) {
          LOG_OUT << "Record node input Storage: " << storage;
          (void)recordedStorages.insert(storage);
          auto storageToOwnerIter = storageToOwner.find(storage);
          CHECK_IF_FAIL(storageToOwnerIter != storageToOwner.end());
          auto *ownerNode = storageToOwnerIter->second;
          CHECK_IF_NULL(ownerNode);
          if (ownerNode->op != ops::Op_End) {
            (void)storagesToFree[currentNode].emplace_back(storage);
          }
        }
      });
    }

    // Current output should freed by itself if it is not freed by later nodes.
    ir::VisitAllTensors(currentNode->output, [&](const ir::TensorPtr &tensor) {
      auto storage = tensor->GetStorage().get();
      CHECK_IF_NULL(storage);
      if (recordedStorages.find(storage) == recordedStorages.end()) {
        LOG_OUT << "Record node output Storage: " << storage;
        (void)recordedStorages.insert(storage);
        auto storageToOwnerIter = storageToOwner.find(storage);
        CHECK_IF_FAIL(storageToOwnerIter != storageToOwner.end());
        auto *ownerNode = storageToOwnerIter->second;
        CHECK_IF_NULL(ownerNode);
        if (ownerNode->op != ops::Op_End) {
          (void)storagesToFree[currentNode].emplace_back(storage);
        }
      }
    });

    // Check if this node owns any of its output storages and record them for allocation
    ir::VisitAllTensors(currentNode->output, [&](const ir::TensorPtr &tensor) {
      auto storage = tensor->GetStorage().get();
      CHECK_IF_NULL(storage);
      // If the storage's owner is the current node, record it for allocation
      if (storageToOwner[storage] == currentNode) {
        (void)storagesToAlloc[currentNode].emplace_back(storage);
      }
    });
  }

  // Step 3.
  for (auto &item : storagesToFree) {
    auto &node = item.first;
    auto &storages = item.second;
    auto iter = nodeToOpRunner_.find(node);
    if (iter == nodeToOpRunner_.end()) {
      LOG_EXCEPTION << "Can not find OpRunner for op: " << ops::ToStr(node->op);
    }
    auto *opRunner = iter->second;
    opRunner->SetStoragesToFree(std::move(storages));
  }

  for (auto &item : storagesToAlloc) {
    auto &node = item.first;
    auto &storages = item.second;
    auto iter = nodeToOpRunner_.find(node);
    if (iter == nodeToOpRunner_.end()) {
      LOG_EXCEPTION << "Can not find OpRunner for op: " << ops::ToStr(node->op);
    }
    auto *opRunner = iter->second;
    opRunner->SetStoragesToAlloc(std::move(storages));
  }
}

void Builder::CreateOpRunners() {
  const size_t nodeNum = graph_->nodes.size();
  opRunners_ = std::make_shared<std::vector<OpRunner>>();
  opRunners_->reserve(nodeNum);
  for (auto &node : graph_->nodes) {
    CHECK_IF_NULL(node);
    if (IsSkipBuildDAKernel(node)) {
      continue;
    }

    auto device = GetOpDeviceType(node);
    auto operatorPtr = ops::CreateOperator(ops::ToStr(node->op), device.type);
    if (operatorPtr == nullptr) {
      LOG_EXCEPTION << "Create operator for: " << ops::ToStr(node->op)
                    << " failed, please register it on platform: " << hardware::GetDeviceNameByType(device.type);
    }
    std::vector<const ir::Value *> inputs;
    for (auto &input : node->inputs) {
      const ir::Value *value = input != nullptr ? input->output.get() : nullptr;
      (void)inputs.emplace_back(value);
    }
    operatorPtr->Init(inputs, node->output.get());

    device::DeviceContext *deviceContext;
    if (auto iter = deviceContexts_.find(device.type); iter != deviceContexts_.end()) {
      deviceContext = iter->second;
    } else {
      auto deviceId = mrt::collective::CollectiveManager::Instance().local_rank_id();
      device::DeviceContextKey deviceContextKey = {hardware::GetDeviceNameByType(device.type), deviceId};
      deviceContext = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(deviceContextKey);
      CHECK_IF_NULL(deviceContext);
      (void)deviceContexts_.emplace(device.type, deviceContext);
    }
    void *stream = deviceContext->deviceResManager_->GetCurrentStream();
    if (device.type != hardware::DeviceType::CPU) {
      CHECK_IF_NULL(stream);
    }
    // TODO: need to support ascend stream creation and getting real dynamic shape info.  // NOLINT(readability/todo)
    (void)(opRunners_->emplace_back(node->op, node->inputs, node->output, std::move(operatorPtr), stream, device,
                                    deviceContext, true /*isDynamicShape*/));
    nodeToOpRunner_.emplace(node.get(), &(opRunners_->back()));
  }
}
}  // namespace runtime
}  // namespace mrt
