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

#include "common/common.h"
#include "tensor/tensor.h"

namespace da {
namespace runtime {
using namespace da::tensor;
constexpr size_t MAX_MEM_SIZE = 1024 * 1024 * 1024 * 4UL;

class MemoryPool {
public:
  using memoryFreeFunc = std::function<void(void *)>;

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

  void Free(DATensor *tensor) const;

private:
  size_t memUsed{0};
  u_char memPool[MAX_MEM_SIZE];
  std::mutex allocMutex_;
  memoryFreeFunc freeFunc_;
};

class TensorDataRecycler {
public:
  explicit TensorDataRecycler(MemoryPool *memPool) : memPool_(memPool) {}
  ~TensorDataRecycler() = default;

  void ForwardRecordInputsRefCounts(DATensor *node);
  void DecreaseInputsRefCounts(DATensor *node);
  void PrintRefCountInfo() const;
  void CheckRefCountInfo() const;

protected:
  void IncreaseInner(DATensor *tensor);
  void DecreaseInner(DATensor *tensor);
  void AppendNodeRefRelations(DATensor *dst, DATensor *src);

private:
  MemoryPool *memPool_;
  std::unordered_map<DATensor *, size_t> refCounts_;
  std::unordered_map<DATensor *, std::vector<DATensor *>> refRelations_;
};

} // namespace runtime
} // namespace da
#endif // __RUNTIME_MEMPOOL_H__
