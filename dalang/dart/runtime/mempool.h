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
#include <mutex>
#include <unordered_set>

#include "common/common.h"

namespace da {
namespace runtime {
constexpr size_t MAX_MEM_SIZE = 1024 * 1024 * 1024 * 4UL;

class MemoryPool {
 public:
  MemoryPool() = default;
  ~MemoryPool() = default;

  void Reset() { memUsed = 0; }

  void *Allocate(size_t size) {
    std::lock_guard<std::mutex> lock(allocMutex_);

    auto newSize = memUsed + size;
    CHECK_IF_FAIL(newSize < MAX_MEM_SIZE);

    void *ptr = memPool + memUsed;
    memUsed = newSize;

    return ptr;
  }

 private:
  size_t memUsed{0};
  u_char memPool[MAX_MEM_SIZE];
  std::mutex allocMutex_;
};

}  // namespace runtime
}  // namespace da
#endif  // __RUNTIME_MEMPOOL_H__
