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

#include <vector>

#include "runtime/mempool.h"
#include "runtime/utils.h"

namespace da {
namespace runtime {
void MemoryPool::Free(DATensor *tensor) const {
  CHECK_IF_NULL(tensor);
  CHECK_IF_NULL(tensor->data);
  if (tensor->type == Type_Tensor) {
    auto **tensorList = reinterpret_cast<DATensor **>(tensor->data);
    CHECK_IF_NULL(tensorList);
    for (size_t i = 0; i < tensor->shape[0]; ++i) {
      CHECK_IF_NULL(tensorList[i]->data);
      freeFunc_(tensorList[i]->data);
    }
  } else {
    freeFunc_(tensor->data);
  }
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

void TensorDataRecycler::ForwardRecordInputsRefCounts(DATensor *node) {
  CHECK_IF_NULL(node);
  if (IsSkipRecordRefCount(node)) {
    return;
  }
  for (size_t i = 0; i < node->inputSize; ++i) {
    if (IsSkipRecordRefCount(node->input[i])) {
      continue;
    }
    CHECK_IF_NULL(node->input[i]);
    AppendNodeRefRelations(node, node->input[i]);
  }
}

void TensorDataRecycler::AppendNodeRefRelations(DATensor *dst, DATensor *src) {
  CHECK_IF_NULL(dst);
  CHECK_IF_NULL(src);
  std::vector<DATensor *> relations;
  if (IsDummyDATensorNode(src)) {
    for (auto related : refRelations_[src]) {
      (void)relations.emplace_back(related);
    }
  } else {
    (void)relations.emplace_back(src);
  }
  for (auto relation : relations) {
    if (!IsDummyDATensorNode(dst)) {
      IncreaseInner(relation);
    }
    (void)refRelations_[dst].emplace_back(relation);
  }
}

void TensorDataRecycler::FreeUnusedNodes(DATensor *node) {
  CHECK_IF_NULL(node);
  if (node->op == ops::Op_return || IsSkipRecordRefCount(node) ||
      IsDummyDATensorNode(node)) {
    return;
  }
  if (nodeReleaseList_.find(node) != nodeReleaseList_.end()) {
    CHECK_IF_NULL(memPool_);
    for (auto releasedNode : nodeReleaseList_[node]) {
      memPool_->Free(releasedNode);
    }
    return;
  }
  for (auto related : refRelations_[node]) {
    DecreaseInner(related);
    if (refCounts_[related] == 0) {
      nodeReleaseList_[node].emplace_back(related);
    }
  }
  (void)nodeReleaseList_[node];
}

void TensorDataRecycler::IncreaseInner(DATensor *tensor) {
  CHECK_IF_NULL(tensor);
  LOG_OUT << "Increase refCount for ops." << ops::ToStr(tensor->op)
          << ", DATensor: " << tensor;
  if (auto iter = refCounts_.find(tensor); iter != refCounts_.end()) {
    iter->second++;
  } else {
    refCounts_[tensor] = 1;
  }
}

void TensorDataRecycler::DecreaseInner(DATensor *tensor) {
  CHECK_IF_NULL(tensor);
  CHECK_IF_FAIL(refCounts_.count(tensor) != 0);
  CHECK_IF_FAIL(refCounts_[tensor] > 0);
  LOG_OUT << "Decrease refCount for ops." << ops::ToStr(tensor->op)
          << ", DATensor: " << tensor;
  refCounts_[tensor]--;
  if (refCounts_[tensor] == 0) {
    LOG_OUT << "Free memory of ops." << ops::ToStr(tensor->op)
            << ", DATensor: " << tensor
            << ", is DATensorList: " << (tensor->type == Type_Tensor);
    memPool_->Free(tensor);
  }
}

void TensorDataRecycler::PrintRefCountInfo() const {
  for (auto refCount : refCounts_) {
    LOG_OUT << "ops." << ops::ToStr(refCount.first->op)
            << ", DATensor: " << refCount.first
            << ", refCount: " << refCount.second;
  }
}

void TensorDataRecycler::CheckRefCountInfo() const {
  for (auto refCount : refCounts_) {
    if (refCount.second != 0) {
      LOG_ERROR << "ops." << ops::ToStr(refCount.first->op)
                << ", DATensor: " << refCount.first
                << ", refCount: " << refCount.second
                << ", there may be memory leak";
    }
  }
}
} // namespace runtime
} // namespace da
