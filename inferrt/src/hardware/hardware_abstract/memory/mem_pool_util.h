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

#ifndef INFERRT_SRC_HARDWARE_MEMORY_MEM_POOL_UTIL_H_
#define INFERRT_SRC_HARDWARE_MEMORY_MEM_POOL_UTIL_H_

#include <atomic>
#include <string>

#include "common/visible.h"

namespace mrt {
namespace memory {
namespace mem_pool {
enum class MemType : int {
  kWeight = 0,
  kConstantValue,
  kKernel,
  kGraphOutput,
  kSomas,
  kSomasOutput,
  kGeConst,
  kGeFixed,
  kBatchMemory,
  kContinuousMemory,
  kPyNativeInput = 10,
  kPyNativeOutput,
  kWorkSpace,
  kOther
};

class MRT_EXPORT Lock {
 public:
  inline void lock() {
    while (locked.test_and_set(std::memory_order_acquire)) {
    }
  }
  inline void unlock() { locked.clear(std::memory_order_release); }

 protected:
  std::atomic_flag locked = ATOMIC_FLAG_INIT;
};

class MRT_EXPORT LockGuard {
 public:
  explicit LockGuard(const Lock &lock) : lock_(const_cast<Lock *>(&lock)) { lock_->lock(); }
  ~LockGuard() { lock_->unlock(); }

 private:
  Lock *lock_;
};

MRT_EXPORT std::string MemTypeToStr(MemType mem_type);

constexpr size_t kPoolGrowSize = 1 << 20;

template <class T>
class ObjectPool {
  struct Buf {
    Buf *next_;
  };

  class Buffer {
    static const std::size_t bucket_size = sizeof(T) > sizeof(Buf) ? sizeof(T) : sizeof(Buf);
    static const std::size_t kDataBucketSize = bucket_size * kPoolGrowSize;

   public:
    explicit Buffer(Buffer *next) : next_(next) {}

    T *GetBlock(std::size_t index) {
      if (index >= kPoolGrowSize) {
        throw std::bad_alloc();
      }
      return reinterpret_cast<T *>(&data_[bucket_size * index]);
    }

    Buffer *const next_;

   private:
    uint8_t data_[kDataBucketSize];
  };

  Buf *free_list_ = nullptr;
  Buffer *buffer_head_ = nullptr;
  std::size_t buffer_index_ = kPoolGrowSize;

 public:
  ObjectPool() = default;
  ObjectPool(ObjectPool &&object_pool) = delete;
  ObjectPool(const ObjectPool &object_pool) = delete;
  ObjectPool operator=(const ObjectPool &object_pool) = delete;
  ObjectPool operator=(ObjectPool &&object_pool) = delete;

  ~ObjectPool() {
    while (buffer_head_ != nullptr) {
      Buffer *buffer = buffer_head_;
      buffer_head_ = buffer->next_;
      delete buffer;
    }
  }

  T *Borrow() {
    if (free_list_ != nullptr) {
      Buf *buf = free_list_;
      free_list_ = buf->next_;
      return reinterpret_cast<T *>(buf);
    }

    if (buffer_index_ >= kPoolGrowSize) {
      buffer_head_ = new Buffer(buffer_head_);
      buffer_index_ = 0;
    }

    return buffer_head_->GetBlock(buffer_index_++);
  }

  void Return(T *obj) {
    Buf *buf = reinterpret_cast<Buf *>(obj);
    buf->next_ = free_list_;
    free_list_ = buf;
  }
};

// Not support older windows version.
template <class T>
class PooledAllocator : private ObjectPool<T> {
 public:
  typedef std::size_t size_type;
  typedef std::ptrdiff_t difference_type;
  typedef T *pointer;
  typedef const T *const_pointer;
  typedef T &reference;
  typedef const T &const_reference;
  typedef T value_type;

  template <class U>
  struct rebind {
    typedef PooledAllocator<U> other;
  };

  pointer allocate(size_type n, const void *hint = 0) {
    if (n != 1 || hint) throw std::bad_alloc();
    return ObjectPool<T>::Borrow();
  }

  void deallocate(pointer p, size_type n) { ObjectPool<T>::Return(p); }

  void construct(pointer p, const_reference val) { new (p) T(val); }

  void destroy(pointer p) { p->~T(); }
};
}  // namespace mem_pool
}  // namespace memory
}  // namespace mrt
#endif
