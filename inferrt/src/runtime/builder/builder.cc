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
std::unique_ptr<Executor> Builder::BuildExecutor() {
  std::unordered_map<ir::Node *, std::vector<size_t>> tensorFreePoint;
  // 1. Analyse ref count
  RecordTensorFreePoint(&tensorFreePoint);
  // 2. Creater OpRunner
  CreateOpRunners(&tensorFreePoint);

  return std::make_unique<Executor>(opRunners_);
}

void Builder::RecordTensorFreePoint(std::unordered_map<ir::Node *, std::vector<size_t>> *tensorFreePoint) const {
  if (graph_ == nullptr || graph_->nodes.empty()) {
    return;
  }

  // TODO:
  // 1. input maybe a tuple, such as AddN's input value.
  // 2. maybe a output tensor has no user, such as tuple of output, only part of tuple have user or return as graph
  // output.
  // 3. need adapt ref, view and heter(cpu) op.

  // Record the last consumer of each node's output
  std::unordered_set<ir::Node *> nodeHasRecorded;
  // Traverse in reverse execution order
  for (size_t i = graph_->nodes.size(); i > 0; --i) {
    size_t nodeIndex = i - 1;
    ir::Node *currentNode = graph_->nodes[nodeIndex].get();
    CHECK_IF_NULL(currentNode)
    // Skip graph input node.
    if (currentNode->op == ops::Op_End) {
      continue;
    }
    // Traverse all inputs of current node
    auto inputNum = currentNode->inputs.size();
    for (size_t inputIdx = 0; inputIdx < inputNum; ++inputIdx) {
      ir::Node *inputNode = currentNode->inputs[inputIdx].get();
      CHECK_IF_NULL(inputNode)
      // Skip input node is graph input case.
      if (inputNode->op == ops::Op_End) {
        continue;
      }

      // First encounter, meaning current node is the last consumer
      if (nodeHasRecorded.find(inputNode) == nodeHasRecorded.end()) {
        (void)nodeHasRecorded.insert(inputNode);
        (void)((*tensorFreePoint)[currentNode].emplace_back(inputIdx));
      }
    }
  }
}

void Builder::CreateOpRunners(std::unordered_map<ir::Node *, std::vector<size_t>> *tensorFreePoint) {
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
    auto iter = tensorFreePoint->find(node.get());
    if (iter != tensorFreePoint->end()) {
      opRunners_->back().SetInputFreeIndex(std::move(iter->second));
    }
  }
}
}  // namespace runtime
}  // namespace mrt
