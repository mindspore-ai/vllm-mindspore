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

#ifndef __RUNTIME_MEMPOOL_H__
#define __RUNTIME_MEMPOOL_H__

#include <cstdlib>
#include <functional>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <utility>

#include "common/common.h"
#include "ir/graph.h"

namespace mrt {
namespace runtime {
constexpr size_t MAX_MEM_SIZE = 1024 * 1024 * 1024 * 4UL;
using memoryFreeFunc = std::function<void(void *)>;

class MemoryPool {
 public:
  MemoryPool() = default;
  ~MemoryPool() = default;

  void Reset() { memUsed = 0; }

  void SetFreeFunc(memoryFreeFunc &&func) { freeFunc_ = func; }

  void *Allocate(size_t size) {
    std::lock_guard<std::mutex> lock(allocMutex_);

    auto newSize = memUsed + size;
    CHECK_IF_FAIL(newSize < MAX_MEM_SIZE);

    void *ptr = memPool + memUsed;
    memUsed = newSize;

    return ptr;
  }

  void Free(ir::NodePtr tensor) const;

 private:
  size_t memUsed{0};
  u_char memPool[MAX_MEM_SIZE];
  std::mutex allocMutex_;
  memoryFreeFunc freeFunc_ = free;
};

class TensorDataRecycler {
 public:
  TensorDataRecycler();
  ~TensorDataRecycler();

  void ForwardRecordInputsRefCounts(ir::NodePtr node);
  void FreeUnusedNodes(ir::NodePtr node);
  void PrintRunningRefCounts() const;
  void ResetRunningRefCounts() { runningRefCounts_ = refCounts_; }

  void SetFreeFunc(memoryFreeFunc &&func) {
    CHECK_IF_NULL(memPool_);
    memPool_->SetFreeFunc(std::move(func));
  }

 protected:
  void IncreaseInner(ir::NodePtr tensor);
  void DecreaseInner(ir::NodePtr tensor);
  void AppendNodeRefRelations(ir::NodePtr dst, ir::NodePtr src);

 private:
  MemoryPool *memPool_{nullptr};
  std::mutex runningRefCountsMutex_;
  std::unordered_map<ir::NodePtr, size_t> runningRefCounts_;
  std::unordered_map<ir::NodePtr, size_t> refCounts_;
  std::unordered_map<ir::NodePtr, std::vector<ir::NodePtr>> refRelations_;
};

}  // namespace runtime
}  // namespace mrt
#endif  // __RUNTIME_MEMPOOL_H__
