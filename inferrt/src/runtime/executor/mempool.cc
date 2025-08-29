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

#include <vector>

#include "runtime/executor/mempool.h"
#include "runtime/utils/utils.h"

namespace mrt {
namespace runtime {
void MemoryPool::Free(ir::NodePtr tensor) const {
  CHECK_IF_NULL(tensor);
  tensor->output = ir::MakeIntrusive<ir::Value>();
}

TensorDataRecycler::TensorDataRecycler() {
  memPool_ = new MemoryPool();
  CHECK_IF_NULL(memPool_);
}

TensorDataRecycler::~TensorDataRecycler() {
  CHECK_IF_NULL(memPool_);
  delete memPool_;
  memPool_ = nullptr;
}

void TensorDataRecycler::ForwardRecordInputsRefCounts(ir::NodePtr node) {
  CHECK_IF_NULL(node);
  if (IsSkipRecordRefCount(node)) {
    return;
  }
  for (auto input : node->inputs) {
    if (IsSkipRecordRefCount(input)) {
      continue;
    }
    CHECK_IF_NULL(input);
    AppendNodeRefRelations(node, input);
  }
}

void TensorDataRecycler::AppendNodeRefRelations(ir::NodePtr dst, ir::NodePtr src) {
  CHECK_IF_NULL(dst);
  CHECK_IF_NULL(src);
  std::vector<ir::NodePtr> relations;
  if (IsDummyNode(src)) {
    for (auto related : refRelations_[src]) {
      (void)relations.emplace_back(related);
    }
  } else {
    (void)relations.emplace_back(src);
  }
  for (auto relation : relations) {
    if (!IsDummyNode(dst)) {
      IncreaseInner(relation);
    }
    (void)refRelations_[dst].emplace_back(relation);
  }
}

void TensorDataRecycler::FreeUnusedNodes(ir::NodePtr node) {
  CHECK_IF_NULL(node);
  if (IsSkipRecordRefCount(node) || IsDummyNode(node)) {
    return;
  }
  for (auto related : refRelations_[node]) {
    DecreaseInner(related);
  }
}

void TensorDataRecycler::IncreaseInner(ir::NodePtr node) {
  CHECK_IF_NULL(node);
  LOG_OUT << "Increase refCount for node: " << node;
  ++refCounts_[node];
}

void TensorDataRecycler::DecreaseInner(ir::NodePtr node) {
  CHECK_IF_NULL(node);
  std::lock_guard<std::mutex> lock(runningRefCountsMutex_);
  CHECK_IF_FAIL(runningRefCounts_.count(node) != 0);
  CHECK_IF_FAIL(runningRefCounts_[node] > 0);
  LOG_OUT << "Decrease refCount for node: " << node;
  --runningRefCounts_[node];
  if (runningRefCounts_[node] == 0) {
    CHECK_IF_NULL(memPool_);
    LOG_OUT << "Free memory of node: " << node;
    memPool_->Free(node);
  }
}

void TensorDataRecycler::PrintRunningRefCounts() const {
  for (auto refCount : runningRefCounts_) {
    LOG_OUT << "node: " << refCount.first << ", refCount: " << refCount.second;
  }
}

}  // namespace runtime
}  // namespace mrt
