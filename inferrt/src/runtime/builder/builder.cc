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
#include "runtime/executor/executor.h"
#include "ops/op_register.h"

namespace mrt {
namespace runtime {

namespace {
void recurseTensorValue(const ir::ValuePtr &value, const std::function<void(const ir::ValuePtr &)> &func) {
  if (value->IsTensor()) {
    func(value);
  } else if (value->IsTuple()) {
    for (auto &item : *value->ToTuple()) {
      recurseTensorValue(item, func);
    }
  }
}
}  // namespace

std::unique_ptr<Executor> Builder::BuildExecutor() {
  RecordStorageFreePoint();
  CreateOpRunners();
  return std::make_unique<Executor>(opRunners_);
}

void Builder::RecordStorageFreePoint() {
  if (graph_ == nullptr || graph_->nodes.empty()) {
    return;
  }

  std::unordered_set<ir::StoragePtr> recordedStorages;

  // The output of graph should not be freed by any node.
  auto &graphOutput = graph_->nodes.back()->output;
  recurseTensorValue(graphOutput, [&](const ir::ValuePtr &tensorValue) {
    auto storage = tensorValue->ToTensor()->GetStorage().get();
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
      recurseTensorValue(inputNode->output, [&](const ir::ValuePtr &tensorValue) {
        auto storage = tensorValue->ToTensor()->GetStorage().get();
        CHECK_IF_NULL(storage);
        if (!storage->CheckOwnsData()) {
          LOG_OUT << "Skip storage that is not managed by mrt";
          return;
        }
        // First encounter, meaning current node is the last consumer
        // and is responsible for freeing the storage.
        if (recordedStorages.find(storage) == recordedStorages.end()) {
          LOG_OUT << "Record node input Storage: " << storage;
          (void)recordedStorages.insert(storage);
          (void)storagesToFree_[currentNode].emplace_back(storage);
        }
      });
    }

    // Current output should freed by itself if it is not freed by later nodes.
    recurseTensorValue(currentNode->output, [&](const ir::ValuePtr &tensorValue) {
      auto storage = tensorValue->ToTensor()->GetStorage().get();
      CHECK_IF_NULL(storage);
      if (recordedStorages.find(storage) == recordedStorages.end()) {
        LOG_OUT << "Record node output Storage: " << storage;
        (void)recordedStorages.insert(storage);
        (void)storagesToFree_[currentNode].emplace_back(storage);
      }
    });
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
    auto operatorPtr = ops::OpFactory<ops::Operator, ops::CPUOpFactory>::GetInstance().Create(ops::ToStr(node->op));
    CHECK_IF_NULL(operatorPtr);
    // TODO: need to support ascend stream creation and getting real dynamic shape info.
    (void)(opRunners_->emplace_back(node.get(), std::move(operatorPtr), nullptr /*stream*/, true /*isDynamicShape*/));
    auto iter = storagesToFree_.find(node.get());
    if (iter != storagesToFree_.end()) {
      opRunners_->back().SetStoragesToFree(std::move(iter->second));
    }
  }
}
}  // namespace runtime
}  // namespace mrt
