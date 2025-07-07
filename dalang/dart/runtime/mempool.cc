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

#include "runtime/mempool.h"

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

void TensorDataRecycler::IncreaseInputsRefCounts(DATensor *node) {
  CHECK_IF_NULL(node);
  LOG_OUT << "Increase refCount for ops." << ops::ToStr(node->op)
          << ", DATensor: " << node;
  for (size_t i = 0; i < node->inputSize; ++i) {
    // Skip weight and const node
    if (IsDATensorConst(node->input[i])) {
      continue;
    }
    CHECK_IF_NULL(node->input[i]);
    IncreaseInner(node->input[i]);
    if (IsDATensorOutputFromInput(node->input[i])) {
      IncreaseInputsRefCounts(node->input[i]);
    }
  }
}

void TensorDataRecycler::DecreaseInputsRefCounts(DATensor *node) {
  CHECK_IF_NULL(node);
  LOG_OUT << "Decrease refCount for ops." << ops::ToStr(node->op)
          << ", DATensor: " << node;
  for (size_t i = 0; i < node->inputSize; ++i) {
    // Skip weight and const node
    if (IsDATensorConst(node->input[i])) {
      continue;
    }
    CHECK_IF_NULL(node->input[i]);
    DecreaseInner(node->input[i]);
    if (IsDATensorOutputFromInput(node->input[i])) {
      DecreaseInputsRefCounts(node->input[i]);
    }
  }
}

void TensorDataRecycler::IncreaseInner(DATensor *tensor) {
  CHECK_IF_NULL(tensor);
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
  refCounts_[tensor]--;
  if (refCounts_[tensor] == 0 && !IsDATensorOutputFromInput(tensor)) {
    LOG_OUT << "Free memory of ops." << ops::ToStr(tensor->op)
            << ", DATensor: " << tensor
            << ", is DATensorList: " << (tensor->type == Type_Tensor);
    memPool_->Free(tensor);
  }
}
} // namespace runtime
} // namespace da
