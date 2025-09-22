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

#ifndef __RUNTIME_UTILS_SPINLOCK_H__
#define __RUNTIME_UTILS_SPINLOCK_H__

#include <atomic>

namespace mrt {
namespace runtime {
class SpinLock {
 public:
  void lock() {
    while (locked_.test_and_set(std::memory_order_acquire)) {
    }
  }

  void unlock() { locked_.clear(std::memory_order_release); }

 private:
  std::atomic_flag locked_ = ATOMIC_FLAG_INIT;
};
}  // namespace runtime
}  // namespace mrt

#endif  // __RUNTIME_UTILS_SPINLOCK_H__
