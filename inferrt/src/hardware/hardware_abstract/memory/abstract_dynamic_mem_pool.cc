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

#include "hardware/hardware_abstract/memory/abstract_dynamic_mem_pool.h"

#include <stdio.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <numeric>

#include "hardware/hardware_abstract/memory/mem_pool_util.h"
#include "common/logger.h"
#include "common/common.h"

namespace mrt {
namespace device {
MemBuf::MemBuf(size_t size, void *addr, uint32_t stream_id, MemBlock *mem_block, MemBufStatus status)
    : prev_(nullptr),
      next_(nullptr),
      size_(size),
      addr_(addr),
      streamId_(stream_id),
      memBlock_(mem_block),
      status_(status) {}

MemBuf::~MemBuf() {}

MemBufAllocator::~MemBufAllocator() {
  LOG_OUT << "MemBufAllocator[" << this << "] : " << BriefInfo() << " deconstruct.";
  for (auto &mem_block : memBlocks_) {
    delete mem_block;
  }
  memBlocks_.clear();
  for (auto mem_buf : freeMemBufs_) {
    delete mem_buf;
  }
  freeMemBufs_.clear();
  for (auto mem_buf : eagerFreeMemBufs_) {
    delete mem_buf;
  }
  eagerFreeMemBufs_.clear();
  delete searchKey_;
}

void MemBufAllocator::ReleaseDeviceRes() {
  LOG_OUT << "Release device resource for allocator, " << BriefInfo() << ", memBlocks_ size : " << memBlocks_.size()
          << ".";
  for (auto mem_block : memBlocks_) {
    LOG_OUT << "Clean mem block : " << mem_block->ToJson() << ".";
    (void)memBlockCleaner_(mem_block);
  }
  for (auto mem_block : memBlocks_) {
    LOG_OUT << "Delete mem block : " << mem_block->ToJson() << ".";
    delete mem_block;
  }
  memBlocks_.clear();

  LOG_OUT << "Free mem buf size : " << freeMemBufs_.size() << ".";
  for (auto mem_buf : freeMemBufs_) {
    delete mem_buf;
  }
  freeMemBufs_.clear();

  LOG_OUT << "Eager free mem buf size : " << eagerFreeMemBufs_.size() << ".";
  for (auto mem_buf : eagerFreeMemBufs_) {
    delete mem_buf;
  }
  eagerFreeMemBufs_.clear();
}

MemBuf *MemBufAllocator::Malloc(size_t size) {
  // Malloc with expand block first.
  if (MS_UNLIKELY(memBlocks_.empty())) {
    return MallocExpandBlock(size);
  }

  searchKey_->size_ = size;
  auto it = freeMemBufs_.lower_bound(searchKey_);
  MemBuf *candidate = nullptr;
  // 1. Try to find in free mem bufs.
  if (MS_LIKELY(it != freeMemBufs_.end())) {
    candidate = *it;
    (void)freeMemBufs_.erase(it);
    return MapAndSplitMemBuf(candidate, size);
  }
  // 2. Try to search available buf, free and eager free buf.
  candidate = SearchAvailableMemBuf(size);
  if (MS_UNLIKELY(candidate != nullptr)) {
    return candidate;
  }
  // 3. Try to find in eager free mem bufs.
  it = eagerFreeMemBufs_.lower_bound(searchKey_);
  if (it != eagerFreeMemBufs_.end()) {
    candidate = *it;
    (void)eagerFreeMemBufs_.erase(it);
    return MapAndSplitMemBuf(candidate, size);
  }

  return nullptr;
}

inline MemBuf *MemBufAllocator::SearchAvailableMemBuf(size_t size) {
  if (!enableEagerFree_ || MS_UNLIKELY(isCustomized_)) {
    return nullptr;
  }
  // Search from back to front, because the free mem buf is sorted by size.
  // More efficient way is to search more candidates, do it in the next version.
  for (auto backward_it = freeMemBufs_.rbegin(); backward_it != freeMemBufs_.rend(); backward_it++) {
    auto mem_buf = *backward_it;
    auto next_buf = mem_buf->next_;
    if (next_buf == nullptr || next_buf->status_ != MemBufStatus::kMemBufEagerFree ||
        mem_buf->size_ + next_buf->size_ < size) {
      continue;
    }

    // Located candidates, try map and split.
    auto need_map_size = size - mem_buf->size_;
    auto mapped_size = memMapper_(need_map_size, next_buf->addr_);
    if (mapped_size != need_map_size) {
      LOG_OUT << "Map mem buf : " << mem_buf->ToJson() << ", next buf : " << next_buf->ToJson() << ", size : " << size
              << ", need_map_size : " << need_map_size << ", mapped_size : " << mapped_size << " failed.";
      return nullptr;
    }
    // Update mem buf.
    freeMemBufs_.erase(mem_buf);
    mem_buf->size_ = size;
    mem_buf->status_ = MemBufStatus::kMemBufUsed;
    // Remove eager free buf and try update it.
    eagerFreeMemBufs_.erase(next_buf);
    next_buf->addr_ = static_cast<uint8_t *>(next_buf->addr_) + need_map_size;
    next_buf->size_ = next_buf->size_ - need_map_size;
    // If next buf is empty, remove it or update remain eager free mem buf.
    if (next_buf->size_ == 0) {
      mem_buf->next_ = next_buf->next_;
      if (next_buf->next_ != nullptr) {
        next_buf->next_->prev_ = mem_buf;
      }
      delete next_buf;
    } else {
      eagerFreeMemBufs_.insert(next_buf);
    }
    return mem_buf;
  }
  return nullptr;
}

bool MemBufAllocator::Free(MemBuf *mem_buf, MemBufStatus target_status) {
  // Change mem buf status to used by event, and wait for event to free.
  if (MS_UNLIKELY(!mem_buf->IsEventNotUsed())) {
    mem_buf->status_ = MemBufStatus::kMemBufUsedByEvent;
    return false;
  }

  mem_buf->status_ = target_status;
  // Try to merge from prev.
  auto prev_buf = mem_buf->prev_;
  if (MS_LIKELY(prev_buf != nullptr && prev_buf->status_ == target_status)) {
    // Erase prev buf pointer
    auto prev = prev_buf->prev_;
    mem_buf->prev_ = prev;
    if (prev != nullptr) {
      prev->next_ = mem_buf;
    }

    mem_buf->addr_ = prev_buf->addr_;
    mem_buf->size_ += prev_buf->size_;
    if (target_status == MemBufStatus::kMemBufIdle) {
      auto ret = freeMemBufs_.erase(prev_buf);
      if (ret == 0) {
        LOG_ERROR << "Erase mem buf : " << mem_buf->ToJson() << " prev buf " << prev_buf->ToJson() << " failed.";
      }
    } else if (target_status == MemBufStatus::kMemBufEagerFree) {
      auto ret = eagerFreeMemBufs_.erase(prev_buf);
      if (ret == 0) {
        LOG_ERROR << "Erase mem buf : " << mem_buf->ToJson() << " prev buf " << prev_buf->ToJson() << " failed.";
      }
    }
    delete prev_buf;
  }
  // Try to merge from next.
  auto next_buf = mem_buf->next_;
  if (MS_LIKELY(next_buf != nullptr && next_buf->status_ == target_status)) {
    // Erase next buf pointer
    auto next = next_buf->next_;
    mem_buf->next_ = next;
    if (next != nullptr) {
      next->prev_ = mem_buf;
    }

    mem_buf->size_ += next_buf->size_;
    if (target_status == MemBufStatus::kMemBufIdle) {
      auto ret = freeMemBufs_.erase(next_buf);
      if (ret == 0) {
        LOG_ERROR << "Erase next buf : " << next_buf->ToJson() << " failed.";
      }
    } else if (target_status == MemBufStatus::kMemBufEagerFree) {
      auto ret = eagerFreeMemBufs_.erase(next_buf);
      if (ret == 0) {
        LOG_ERROR << "Erase next buf : " << next_buf->ToJson() << " failed.";
      }
    }
    delete next_buf;
  }

  if (target_status == MemBufStatus::kMemBufIdle) {
    (void)freeMemBufs_.emplace(mem_buf);
  } else if (target_status == MemBufStatus::kMemBufEagerFree) {
    (void)eagerFreeMemBufs_.emplace(mem_buf);
  }

  return true;
}

MemBuf *MemBufAllocator::MallocExpandBlock(size_t size) {
  MemBlock *mem_block = ExpandBlock(size);
  if (mem_block == nullptr) {
    return nullptr;
  }
  MemBuf *candidate = new MemBuf(
    mem_block->size_, mem_block->addr_, mem_block->streamId_, mem_block,
    MS_LIKELY(!isCustomized_) && enableEagerFree_ ? MemBufStatus::kMemBufEagerFree : MemBufStatus::kMemBufIdle);
  if (candidate->size_ < size) {
    if (candidate->status_ == MemBufStatus::kMemBufIdle) {
      (void)freeMemBufs_.emplace(candidate);
    } else {
      (void)eagerFreeMemBufs_.emplace(candidate);
    }
    LOG_OUT << "Candidate size: " << candidate->size_ << " is less than required size : " << size << ".";
    return nullptr;
  }

  return MapAndSplitMemBuf(candidate, size);
}

void MemBufAllocator::Initialize(size_t size) {
  LOG_OUT << "Initialize allocator : " << BriefInfo() << " with size : " << size << ".";
  if (enableEagerFree_ || MS_UNLIKELY(isCustomized_)) {
    LOG_OUT << "Skip initialization of allocator, since vmm is enabled.";
    return;
  }
  MemBlock *mem_block = ExpandBlock(size);
  if (mem_block == nullptr) {
    LOG_OUT << "Initialize allocator failed, size : " << size << ".";
    return;
  }
  MemBuf *mem_buf =
    new MemBuf(mem_block->size_, mem_block->addr_, mem_block->streamId_, mem_block, MemBufStatus::kMemBufIdle);
  (void)freeMemBufs_.emplace(mem_buf);
}

const std::pair<size_t, size_t> MemBufAllocator::FreeIdleMemsByEagerFree() {
  // Free all idle mem bufs.
  size_t eager_free_size = 0;
  for (auto mem_buf : freeMemBufs_) {
    eager_free_size += mem_buf->size_;
    Free(mem_buf, MemBufStatus::kMemBufEagerFree);
  }
  freeMemBufs_.clear();
  // Do eager free on eager free mem bufs.
  size_t real_free_size = 0;
  for (auto mem_buf : eagerFreeMemBufs_) {
    LOG_OUT << "Eager free mem buf : " << mem_buf << ", details : " << mem_buf->ToJson() << ".";
    real_free_size += memEagerFreer_(mem_buf->addr_, mem_buf->size_);
  }
  LOG_OUT << "Free idle mems by eager free, eager_free_size : " << eager_free_size
          << ", real_free_size : " << real_free_size << ".";
  return std::make_pair(eager_free_size, real_free_size);
}

size_t MemBufAllocator::ReleaseFreeBlocks() {
  size_t release_size = 0;
  for (auto iter = memBlocks_.begin(); iter != memBlocks_.end();) {
    auto mem_block = *iter;
    MemBuf mem_buf(mem_block->size_, mem_block->addr_, mem_block->streamId_, mem_block, MemBufStatus::kMemBufIdle);
    // Judge if mem block in free mem bufs.
    auto &&it = freeMemBufs_.find(&mem_buf);
    if (it == freeMemBufs_.end()) {
      iter++;
      continue;
    }
    auto mem_buf_it = *it;
    if (mem_buf_it->addr_ == mem_block->addr_ && mem_buf_it->size_ == mem_block->size_) {
      LOG_OUT << "Release mem block : " << mem_block->ToJson() << ".";
      bool ret = memBlockCleaner_(mem_block);
      if (!ret) {
        LOG_OUT << "Clean mem block : " << mem_block->ToJson() << " failed.";
        iter++;
        continue;
      }
      freeMemBufs_.erase(it);
      delete mem_buf_it;
      release_size += mem_block->size_;
      delete mem_block;
      iter = memBlocks_.erase(iter);
    } else {
      iter++;
    }
  }
  return release_size;
}

inline MemBuf *MemBufAllocator::MapAndSplitMemBuf(MemBuf *candidate, size_t size) {
  size_t remaining_size = candidate->size_ - size;
  // Mmap memory first.
  if (candidate->status_ == MemBufStatus::kMemBufEagerFree) {
    size_t map_size = (remaining_size >= kDynamicMemAlignSize) ? size : candidate->size_;
    auto mapped_size = memMapper_(map_size, candidate->addr_);
    if (mapped_size != map_size) {
      LOG_OUT << "Mapped_size : " << mapped_size << " is not equal to required size : " << map_size
              << ", mem buf info : " << candidate->ToJson() << ".";
      (void)eagerFreeMemBufs_.emplace(candidate);
      return nullptr;
    }
  }

  bool need_split = remaining_size >= kDynamicMemAlignSize;
  // Try to split mem buf.
  if (MS_LIKELY(need_split)) {
    void *remaining_addr = static_cast<uint8_t *>(candidate->addr_) + size;
    auto remaining_buf =
      new MemBuf(remaining_size, remaining_addr, candidate->streamId_, candidate->memBlock_, candidate->status_);

    auto next = candidate->next_;
    if (next != nullptr) {
      next->prev_ = remaining_buf;
      remaining_buf->next_ = next;
    }
    candidate->next_ = remaining_buf;
    remaining_buf->prev_ = candidate;
    if (remaining_buf->status_ == MemBufStatus::kMemBufIdle) {
      (void)freeMemBufs_.emplace(remaining_buf);
    } else {
      (void)eagerFreeMemBufs_.emplace(remaining_buf);
    }

    // Update candidate size.
    candidate->size_ = size;
  }

  candidate->status_ = MemBufStatus::kMemBufUsed;
  // Update mem block usage.
  candidate->memBlock_->UpdateBorderAddr(candidate);

  return candidate;
}

inline MemBlock *MemBufAllocator::ExpandBlock(size_t size) {
  MemBlock *mem_block = memBlockExpander_(size);
  if (mem_block == nullptr) {
    LOG_OUT << "Expand block failed, expand size : " << size << ", memory is not enough.";
    return nullptr;
  }

  if (mem_block->size_ < size) {
    LOG_OUT << "Expand block failed, expand size : " << mem_block->size_ << " is less than require size : " << size
            << ".";
  }

  (void)memBlocks_.emplace_back(mem_block);
  return mem_block;
}

AbstractDynamicMemPool::AbstractDynamicMemPool() {}

void AbstractDynamicMemPool::Initialize(size_t init_size, size_t increase_size, size_t max_size) {
  if (init_size == 0) {
    LOG_OUT << "Skip initialization of memory pool since init size is not configured.";
    return;
  }

  LockGuard lock(lock_);
  LOG_OUT << "Initialize dynamic memory pool, init size : " << init_size << ", increase size : " << increase_size
          << ", max size : " << max_size << ".";
  initSize_ = init_size >> 1;
  increaseSize_ = increase_size;
  maxSize_ = max_size;

  // Do initialization with init size.
  auto persistent_allocator = GetMemBufAllocator(initSize_, true, kDefaultStreamIndex);
  persistent_allocator->Initialize(AlignMemorySize(initSize_));
  auto common_allocator = GetMemBufAllocator(initSize_, false, kDefaultStreamIndex);
  common_allocator->Initialize(AlignMemorySize(initSize_));
}

void AbstractDynamicMemPool::ReleaseDeviceRes() {
  LockGuard lock(lock_);
  for (const auto &iter : streamPairMemBufs_) {
    auto size = iter.second.size();
    LOG_OUT << "Event referred streamPairMemBufs_[" << iter.first.first << "-" << iter.first.second
            << "], size : " << size << ".";
  }
  // Clear map of address to mem buf.
  for (const auto &iter : addrMemBufAllocators_) {
    auto mem_buf = iter.second.first;
    delete mem_buf;
  }
  addrMemBufAllocators_.clear();

  LOG_OUT << "Release device resource for " << GetMemoryPoolType() << " : " << memStat_.ToReadableString() << ".";
  for (const auto &stream_id_allocator : streamIdAllocators_) {
    const auto &allocator = stream_id_allocator.second;
    allocator->ReleaseDeviceRes();
  }
  for (const auto &customized_allocator : customizedAllocators_) {
    const auto &allocator = customized_allocator.second;
    allocator->ReleaseDeviceRes();
  }
  streamIdAllocators_.clear();
  customizedAllocators_.clear();
  streamPairMemBufs_.clear();
  memStat_.Reset();
}

/**
 * @brief Alloc tensor mem.
 * Allocation follow steps below:
 *    1 align size
 *    2 find from current allocator, if failed transfer to 3
 *    3 find from another allocator, if failed transfer to 4
 *    4 do eager free and find from current allocator again, if failed transfer to 5
 *    5 expand block
 */
DeviceMemPtr AbstractDynamicMemPool::AllocTensorMem(size_t size, bool from_persistent_mem, bool, uint32_t stream_id) {
  size_t align_size = AlignMemorySize(size);
  LockGuard lock(lock_);
  auto &&mem_buf_allocator = AllocMemBuf(align_size, from_persistent_mem, stream_id);
  if (MS_UNLIKELY(mem_buf_allocator.first == nullptr)) {
    return nullptr;
  }

  (void)addrMemBufAllocators_.emplace(mem_buf_allocator.first->addr_, mem_buf_allocator);
  return mem_buf_allocator.first->addr_;
}

/**
 * @brief Alloc mem buf.
 * Strategy when vmm is disable:
 *    Persistent memory:  First malloc form its own pool, if fails, try to malloc from common pool.
 *    Common memory:  First malloc from its own pool, if fails, it will try to expand the pool.
 *                    If the expansion fails, try to malloc from persistent pool.
 */
inline std::pair<MemBuf *, MemBufAllocator *> AbstractDynamicMemPool::AllocMemBuf(size_t align_size,
                                                                                  bool from_persistent_mem,
                                                                                  uint32_t stream_id) {
  auto allocator = GetMemBufAllocator(align_size, from_persistent_mem, stream_id);

  auto mem_buf = allocator->Malloc(align_size);
  if (MS_UNLIKELY(mem_buf == nullptr)) {
    // Enable malloc from another allocator when from_persistent_mem is true and vmm is not enabled.
    if (!enableVmm_ && from_persistent_mem && MS_LIKELY(!enableCustomAllocator_)) {
      auto common_allocator = GetMemBufAllocator(align_size, false, stream_id);
      mem_buf = common_allocator->Malloc(align_size);
      allocator = common_allocator;
    }

    if (MS_UNLIKELY(mem_buf == nullptr)) {
      if ((enableVmm_ || IsEnableEagerFree()) && MS_LIKELY(!enableCustomAllocator_)) {
        WaitPipelineHelper();
        if (!SyncAllStreams()) {
          LOG_ERROR << "Sync all streams failed.";
          return std::make_pair(nullptr, nullptr);
        }
        (void)FreeIdleMemsByEagerFree();
        mem_buf = allocator->Malloc(align_size);
      }
      if (MS_UNLIKELY(mem_buf == nullptr)) {
        mem_buf = allocator->MallocExpandBlock(align_size);
        if (MS_UNLIKELY(mem_buf == nullptr)) {
          if (MS_LIKELY(!from_persistent_mem) && MS_LIKELY(!enableCustomAllocator_)) {
            // Common pool expand block failed, try to malloc from persistent pool.
            auto persistent_allocator = GetMemBufAllocator(align_size, true, stream_id);
            mem_buf = persistent_allocator->Malloc(align_size);
            if (MS_LIKELY(mem_buf != nullptr)) {
              allocator = persistent_allocator;
            }
          }

          if (MS_UNLIKELY(mem_buf == nullptr)) {
            LOG_OUT << "Alloc tensor mem failed and try to sync all events to release memory.";
            (void)DoSyncAllEvents();
            mem_buf = allocator->Malloc(align_size);
            if (MS_UNLIKELY(mem_buf == nullptr)) {
              return std::make_pair(nullptr, nullptr);
            }
          }
        }
      }
    }
  }

  // Update stat.
  memStat_.usedSize_ += mem_buf->size_;
  memStat_.UpdatePeakSize(enableVmm_, GetVmmUsedMemSize());
  return std::make_pair(mem_buf, allocator);
}

std::vector<DeviceMemPtr> AbstractDynamicMemPool::AllocContinuousTensorMem(const std::vector<size_t> &size_list,
                                                                           uint32_t stream_id) {
  std::vector<DeviceMemPtr> device_addr_list;
  size_t total_size = std::accumulate(size_list.begin(), size_list.end(), static_cast<size_t>(0));
  // Pre-alloc the one whole piece memory.
  auto device_addr = AbstractDynamicMemPool::AllocTensorMem(total_size, false, false, stream_id);
  if (device_addr == nullptr) {
    return device_addr_list;
  }

  (void)device_addr_list.emplace_back(device_addr);
  if (size_list.size() == 1) {
    return device_addr_list;
  }

  // Try to split mem bufs.
  LockGuard lock(lock_);
  auto &&it = addrMemBufAllocators_.find(device_addr);
  if (it != addrMemBufAllocators_.end()) {
    auto mem_buf = it->second.first;
    auto allocator = it->second.second;
    mem_buf->size_ = size_list[0];
    MemBuf *prev_mem_buf = mem_buf;
    void *next_addr = static_cast<uint8_t *>(mem_buf->addr_) + size_list[0];
    total_size -= size_list[0];
    for (size_t i = 1; i < size_list.size(); i++) {
      auto new_mem_buf = new MemBuf(size_list[i], next_addr, stream_id, mem_buf->memBlock_, MemBufStatus::kMemBufUsed);
      new_mem_buf->Link(prev_mem_buf, prev_mem_buf->next_);
      (void)addrMemBufAllocators_.emplace(new_mem_buf->addr_, std::make_pair(new_mem_buf, allocator));
      // Update result.
      (void)device_addr_list.emplace_back(next_addr);
      // Update next addr and prev mem buf.
      if (i < size_list.size() - 1) {
        next_addr = static_cast<uint8_t *>(next_addr) + size_list[i];
        total_size -= size_list[i];
        prev_mem_buf = new_mem_buf;
      } else {
        // Update last mem buf
        if (total_size != size_list[i]) {
          LOG_OUT << "Remain size : " << total_size << " is not equal to last size : " << size_list[i] << ".";
          new_mem_buf->size_ = total_size;
        }
      }
    }
  } else {
    // Unreachable routine.
    LOG_ERROR << "Find addr : " << device_addr << " failed.";
  }

  return device_addr_list;
}

// The main program entry of memory free.
void AbstractDynamicMemPool::FreeTensorMem(const DeviceMemPtr &device_addr) {
  LockGuard lock(lock_);
  (void)DoFreeTensorMem(device_addr);
}

// The main program entry of memory free.
bool AbstractDynamicMemPool::DoFreeTensorMem(const DeviceMemPtr &device_addr) {
  void *addr = device_addr;
  auto &&it = addrMemBufAllocators_.find(device_addr);
  if (MS_LIKELY(it != addrMemBufAllocators_.end())) {
    auto allocator = it->second.second;
    auto mem_buf = it->second.first;
    auto free_size = mem_buf->size_;
    if (MS_LIKELY(allocator->Free(mem_buf))) {
      memStat_.usedSize_ -= free_size;
      (void)addrMemBufAllocators_.erase(it);
      return true;
    }
  } else {
    // This may be normal case.
    LOG_OUT << "Free tensor mem failed, can not find address : " << addr << ".";
  }
  return false;
}

inline MemBufAllocator *AbstractDynamicMemPool::GetMemBufAllocator(size_t size, bool from_persistent_mem,
                                                                   uint32_t stream_id) {
  // Not use small pool.
  const AllocatorInfo key{stream_id, from_persistent_mem, false};
  LOG_OUT << "Get allocator, " << key.ToString() << ".";

  MemBufAllocatorPtr allocator = nullptr;

  auto &&it = streamIdAllocators_.find(key);
  if (it == streamIdAllocators_.end()) {
    allocator = GenerateAllocator(key);
    (void)streamIdAllocators_.emplace(key, allocator);
  } else {
    allocator = it->second;
  }
  return allocator.get();
}

// Keep addrs is in free addrs, so here find mem bufs first.
// And then, traverse keep addrs and spilt candidates.
void AbstractDynamicMemPool::FreePartTensorMems(const std::vector<DeviceMemPtr> &free_addrs,
                                                const std::vector<DeviceMemPtr> &keep_addrs,
                                                const std::vector<size_t> &keep_addr_sizes) {
  LOG_OUT << "Free part tensor mems.";
  LockGuard lock(lock_);
  (void)DoFreePartTensorMems(free_addrs, keep_addrs, keep_addr_sizes);
}

std::vector<MemBuf *> AbstractDynamicMemPool::DoFreePartTensorMems(const std::vector<DeviceMemPtr> &free_addrs,
                                                                   const std::vector<DeviceMemPtr> &keep_addrs,
                                                                   const std::vector<size_t> &keep_addr_sizes) {
  std::vector<MemBuf *> mem_bufs;
  std::map<void *, std::pair<MemBuf *, MemBufAllocator *>> candidates;
  for (const auto &free_addr : free_addrs) {
    auto &&it = addrMemBufAllocators_.find(free_addr);
    if (it != addrMemBufAllocators_.end()) {
      (void)candidates.emplace(it->first, it->second);
    } else {
      // This is illegal routine, but level0 case entered.
      LOG_OUT << "Find address : " << free_addr << " failed.";
    }
  }

  std::set<std::uintptr_t> processed_keep_addrs;
  for (size_t i = 0; i < keep_addrs.size(); i++) {
    auto keep_addr = keep_addrs[i];
    std::uintptr_t keep_addr_to_size = reinterpret_cast<std::uintptr_t>(keep_addr);
    if (processed_keep_addrs.count(keep_addr_to_size) > 0) {
      LOG_OUT << "Duplicate keep address : " << keep_addr << ".";
      continue;
    }
    (void)processed_keep_addrs.insert(keep_addr_to_size);
    auto &&it = candidates.upper_bound(keep_addr);
    if (it == candidates.begin()) {
      LOG_OUT << "Locate keep addr : " << keep_addr << " failed.";
      continue;
    }
    auto iter = --it;
    auto mem_buf = iter->second.first;
    auto allocator = iter->second.second;
    std::uintptr_t base_start = reinterpret_cast<std::uintptr_t>(mem_buf->addr_);
    std::uintptr_t base_end = base_start + mem_buf->size_;
    std::uintptr_t keep_start = keep_addr_to_size;
    std::uintptr_t keep_end = keep_start + keep_addr_sizes[i];
    // Since free part tensor mem may double free keep addr, continue for these keep addrs.
    if (keep_start >= base_end) {
      LOG_OUT << "Check range error, base start : " << base_start << ", base end : " << base_end
              << ", keep start : " << keep_start << ", keep end : " << keep_end << ".";
      continue;
    }
    // Split candidates. If keep start equal to base start, split mem buf into two parts, or three parts.
    // First construct keep mem buf and set it into addrMemBufAllocators_, then process head buf and tail buf.
    MemBuf *keep_mem_buf = nullptr;
    if (keep_start == base_start) {
      keep_mem_buf = mem_buf;
      keep_mem_buf->size_ = keep_addr_sizes[i];
      // Remove keep addr since keep start equal to base start, no need to free keep addr any more.
      (void)candidates.erase(mem_buf->addr_);
    } else {
      // Split middle mem buf.
      keep_mem_buf =
        new MemBuf(keep_addr_sizes[i], keep_addr, mem_buf->streamId_, mem_buf->memBlock_, mem_buf->status_);
      keep_mem_buf->Link(mem_buf, mem_buf->next_);
      (void)addrMemBufAllocators_.emplace(keep_addr, std::make_pair(keep_mem_buf, allocator));
      std::uintptr_t prev_remain_size = keep_start - base_start;
      mem_buf->size_ = prev_remain_size;
    }
    (void)mem_bufs.emplace_back(keep_mem_buf);
    LOG_OUT << "keep_mem_buf : " << keep_mem_buf->ToJson() << ".";
    // Process last mem buf.
    if (keep_end < base_end) {
      void *last_addr = static_cast<uint8_t *>(keep_mem_buf->addr_) + keep_mem_buf->size_;
      auto last_mem_buf = new MemBuf(base_end - keep_end, last_addr, keep_mem_buf->streamId_, keep_mem_buf->memBlock_,
                                     mem_buf->status_);
      last_mem_buf->Link(keep_mem_buf, keep_mem_buf->next_);
      (void)addrMemBufAllocators_.emplace(last_mem_buf->addr_, std::make_pair(last_mem_buf, allocator));
      if (candidates.count(last_mem_buf->addr_) > 0) {
        LOG_OUT << "Duplicate address : " << last_mem_buf->addr_ << ".";
      }
      LOG_OUT << "last mem buf : " << last_mem_buf->ToJson() << ".";
      (void)candidates.emplace(last_mem_buf->addr_, std::make_pair(last_mem_buf, allocator));
    }
  }
  for (const auto &candidate : candidates) {
    auto mem_buf = candidate.second.first;
    if (!AbstractDynamicMemPool::DoFreeTensorMem(mem_buf->addr_)) {
      LOG_ERROR << "Free device address failed : " << mem_buf->addr_ << ", mem_buf : " << mem_buf->ToJson() << ".";
    }
  }
  return mem_bufs;
}

MemBufAllocatorPtr AbstractDynamicMemPool::GenerateAllocator(const AllocatorInfo &allocator_key) {
  const auto is_persistent = allocator_key.from_persistent_mem;
  const auto stream_id = allocator_key.stream_id;
  const auto is_small = allocator_key.use_small_pool;

  LOG_OUT << "Generate allocator, " << allocator_key.ToString() << ".";
  std::function<MemBlock *(size_t)> mem_block_expander = [&, is_persistent = is_persistent,
                                                          stream_id = stream_id](size_t size) {
    size_t block_size = CalMemBlockAllocSize(size, is_persistent);
    MemBlock *mem_block = nullptr;
    if (block_size == 0) {
      LOG_OUT << "Malloc mem block failed, is enable eager free : " << IsEnableEagerFree()
              << ", is enable vmm : " << IsEnableVmm() << ", size : " << size << ", block size is  0.";
      return mem_block;
    }
    DeviceMemPtr addr = nullptr;
    size_t alloc_size;
    LOG_OUT << "Malloc mem block, is enable eager free : " << IsEnableEagerFree()
            << ", is enable vmm : " << IsEnableVmm() << ", size : " << size << ", block size : " << block_size << ".";
    if (IsEnableVmm() || IsEnableEagerFree()) {
      // Virtual address is unlimited.
      auto eager_free_size = std::max(block_size, static_cast<size_t>(total_mem_size()));
      alloc_size = AllocDeviceMemByEagerFree(eager_free_size, &addr);
      memStat_.eagerFreeSize_ += alloc_size;
    } else {
      alloc_size = AllocDeviceMem(block_size, &addr);
      if (alloc_size < block_size) {
        LOG_OUT << "Alloc device mem failed, alloc size : " << alloc_size << ", block size : " << block_size << ".";
      }
    }
    if (alloc_size == 0) {
      return mem_block;
    }
    memStat_.allocSize_ += alloc_size;
    mem_block = new MemBlock(alloc_size, addr, stream_id);
    LOG_OUT << "Malloc mem block : " << mem_block->ToJson() << ".";
    return mem_block;
  };

  std::function<bool(MemBlock *)> mem_block_cleaner = [&](MemBlock *mem_block) {
    memStat_.allocSize_ -= mem_block->size_;
    // Call free device mem as ascend memory pool would do stat in free operation.
    return FreeDeviceMem(mem_block->addr_);
  };
  std::function<size_t(size_t size, void *addr)> mem_mapper = [&](size_t size, void *addr) {
    memStat_.eagerFreeSize_ -= size;
    return MmapDeviceMem(size, addr);
  };
  std::function<size_t(void *addr, const size_t size)> mem_eager_freer = [&](void *addr, const size_t size) {
    LOG_OUT << "Eager free addr : " << addr << ", size : " << size << ".";
    return FreeDeviceMemByEagerFree(addr, size);
  };

  return std::make_shared<MemBufAllocator>(mem_block_expander, mem_block_cleaner, mem_mapper, mem_eager_freer,
                                           IsEnableVmm() || IsEnableEagerFree(), is_persistent, stream_id, is_small);
}

// Element in vector : <memory_stream_id, addr>
bool AbstractDynamicMemPool::RecordEvent(int64_t task_id_on_stream, uint32_t user_stream_id,
                                         const std::vector<std::pair<uint32_t, DeviceMemPtr>> &memory_stream_addresses,
                                         const DeviceEventPtr &event) {
  LOG_OUT << "Record event for task id on stream : " << task_id_on_stream << ", user stream id : " << user_stream_id
          << ".";
  LockGuard lock(lock_);
  for (auto &[memory_stream_id, addr] : memory_stream_addresses) {
    auto &&it = addrMemBufAllocators_.find(addr);
    if (it != addrMemBufAllocators_.end()) {
      auto mem_buf = it->second.first;
      if (mem_buf->IsEventNotUsed()) {
        memStat_.usedByEventSize_ += mem_buf->size_;
      }
      LOG_OUT << "Record event for : " << mem_buf->ToJson() << ".";
      (void)mem_buf->RecordEvent(task_id_on_stream, user_stream_id, event);
      (void)streamPairMemBufs_[std::make_pair(user_stream_id, memory_stream_id)].emplace(mem_buf);
    } else {
      // Output of somas sub graph may be used by somas sub graph inner node, address may not be kept in mem pool.
      LOG_OUT << "Unknown address : " << addr << ".";
    }
  }
  return true;
}

bool AbstractDynamicMemPool::WaitEvent(int64_t task_id_on_stream, uint32_t user_stream_id, uint32_t memory_stream_id) {
  LOG_OUT << "Wait event for task id on stream : " << task_id_on_stream << ", user stream id : " << user_stream_id
          << ", memory stream id : " << memory_stream_id << ".";
  LockGuard lock(lock_);
  auto key = std::make_pair(user_stream_id, memory_stream_id);
  auto iter = streamPairMemBufs_.find(key);
  if (iter == streamPairMemBufs_.end()) {
    return false;
  }

  auto mem_bufs_ = iter->second;
  for (const auto &mem_buf : mem_bufs_) {
    LOG_OUT << "Wait event for : " << mem_buf->ToJson() << ".";
    mem_buf->WaitEvent(task_id_on_stream, user_stream_id);
    // Remove event and try to free memory.
    if (mem_buf->IsEventNotUsed()) {
      memStat_.usedByEventSize_ -= mem_buf->size_;
      // Force clear all mem bufs.
      for (auto &stream_pair_mem_bufs : streamPairMemBufs_) {
        (void)stream_pair_mem_bufs.second.erase(mem_buf);
      }
      if (mem_buf->status_ == DynamicMemBufStatus::kMemBufUsedByEvent) {
        (void)DoFreeTensorMem(mem_buf->addr_);
      }
    }
  }
  return true;
}

bool AbstractDynamicMemPool::WaitEvent(int64_t task_id_on_stream, uint32_t memory_stream_id) {
  LOG_OUT << "Wait event for task id on stream : " << task_id_on_stream << ", memory stream id : " << memory_stream_id
          << ".";
  LockGuard lock(lock_);
  for (auto &stream_pair_mem_bufs : streamPairMemBufs_) {
    const auto &[user_stream, memory_stream] = stream_pair_mem_bufs.first;
    if (memory_stream != memory_stream_id) {
      continue;
    }
    auto mem_bufs = stream_pair_mem_bufs.second;
    for (const auto &mem_buf : mem_bufs) {
      LOG_OUT << "Wait event for : " << mem_buf->ToJson() << ".";
      mem_buf->WaitEvent(task_id_on_stream, user_stream);
      // Remove event and try to free memory.
      if (mem_buf->IsEventNotUsed()) {
        memStat_.usedByEventSize_ -= mem_buf->size_;
        // Force clear all mem bufs.
        for (auto &kv : streamPairMemBufs_) {
          (void)kv.second.erase(mem_buf);
        }
        if (mem_buf->status_ == DynamicMemBufStatus::kMemBufUsedByEvent) {
          (void)DoFreeTensorMem(mem_buf->addr_);
        }
      }
    }
  }
  return true;
}

bool AbstractDynamicMemPool::SyncAllEvents() {
  LOG_OUT << "Sync all events, stream_pair_addresses_ size : " << streamPairMemBufs_.size() << ".";
  LockGuard lock(lock_);
  return DoSyncAllEvents();
}

bool AbstractDynamicMemPool::DoSyncAllEvents() {
  if (streamPairMemBufs_.empty()) {
    return false;
  }

  std::set<MemBuf *> carry_event_mem_bufs;
  for (const auto &stream_pair_mem_buf : streamPairMemBufs_) {
    for (const auto &mem_buf : stream_pair_mem_buf.second) {
      (void)carry_event_mem_bufs.emplace(mem_buf);
    }
  }
  for (auto &mem_buf : carry_event_mem_bufs) {
    if (mem_buf->SyncAllEvents() && mem_buf->status_ == DynamicMemBufStatus::kMemBufUsedByEvent) {
      (void)DoFreeTensorMem(mem_buf->addr_);
    }
  }

  streamPairMemBufs_.clear();
  return true;
}

size_t AbstractDynamicMemPool::CalMemBlockAllocSize(size_t size, bool from_persistent_mem, bool) {
  auto device_free_mem_size = free_mem_size();
  // Make sure available mem is enough.
  if (device_free_mem_size < size) {
    LOG_OUT << "Memory not enough: current free memory size[" << device_free_mem_size
            << "] is smaller than required size[" << size << "].";
    return 0;
  }
  auto unit_size = MemAllocUnitSize(from_persistent_mem);
  if (device_free_mem_size < unit_size) {
    LOG_OUT << "Device memory size [" << device_free_mem_size << "] is smaller than unit size [" << unit_size << "].";
  }
  // Calculate alloc size.
  size_t alloc_size = unit_size;
  if (size > unit_size) {
    alloc_size = ((size + unit_size - 1) / unit_size) * unit_size;
  }
  return std::min(alloc_size, device_free_mem_size);
}

void AbstractDynamicMemPool::DefragMemory() {
  LOG_OUT << "Try to defrag memory.";
  LockGuard lock(lock_);

  if (!enableVmm_) {
    LOG_OUT << "Skip defrag memory since vmm is not enabled.";
    return;
  }

  if (eagerFreeCount_ == 0) {
    LOG_OUT << "Exit defrag memory since eager free count is 0.";
    return;
  }
  if (lastEagerFreeCount_ == eagerFreeCount_) {
    LOG_OUT << "Exit defrag memory since last eager free count equals to eager free count : " << lastEagerFreeCount_
            << ".";
    return;
  }

  LOG_OUT << "Defrag memory start.";
  WaitPipelineHelper();
  if (!SyncAllStreams()) {
    LOG_ERROR << "Sync all streams failed.";
    return;
  }
  const auto [eager_free_size, real_free_size] = FreeIdleMemsByEagerFree();
  LOG_OUT << "Defrag memory, eager_free_size : " << eager_free_size << ", real_free_size : " << real_free_size << ".";
  lastEagerFreeCount_ = eagerFreeCount_;
}

void AbstractDynamicMemPool::WaitPipelineHelper() {
  if (pipelineCallback_) {
    lock_.unlock();
    pipelineCallback_();
    lock_.lock();
  }
}

std::string AbstractDynamicMemPool::DynamicMemPoolStateInfo() const {
  std::stringstream ss;
  // Classify mem buf and stat mem buf state info.
  size_t mem_buf_used_stat[static_cast<int>(memory::mem_pool::MemType::kOther) + 1] = {0};
  struct AddrComparator {
    bool operator()(MemBuf *const &left, MemBuf *const &right) const { return left->addr_ < right->addr_; }
  };
  std::map<MemBufAllocator *, std::set<MemBuf *, AddrComparator>> allocator_mem_bufs;
  for (const auto &addr_mem_buf_allocator : addrMemBufAllocators_) {
    const auto allocator = addr_mem_buf_allocator.second.second;
    const auto mem_buf = addr_mem_buf_allocator.second.first;
    mem_buf_used_stat[static_cast<int>(mem_buf->allocType_)] += mem_buf->size_;
    auto &mem_bufs = allocator_mem_bufs[allocator];
    (void)mem_bufs.insert(mem_buf);
  }
  for (const auto &[allocator, mem_bufs] : allocator_mem_bufs) {
    ss << "\tIn used mem buf info for " << allocator->BriefInfo() << ", mem_bufs size : " << mem_bufs.size() << "\n";
  }

  size_t other_used_size = 0;
  int start = static_cast<int>(memory::mem_pool::MemType::kGraphOutput);
  int end = static_cast<int>(memory::mem_pool::MemType::kOther);
  for (int i = start; i <= end; i++) {
    other_used_size += mem_buf_used_stat[i];
  }

  ss << "The dynamic memory pool[" << GetMemoryPoolType() << "] stat info : " << memStat_.ToReadableString()
     << ", actual peak used mem:" << ActualPeakStatistics() / kMBToByte
     << "M. Weight used size:" << mem_buf_used_stat[static_cast<int>(memory::mem_pool::MemType::kWeight)] / kMBToByte
     << "M, constant value used size:"
     << mem_buf_used_stat[static_cast<int>(memory::mem_pool::MemType::kConstantValue)] / kMBToByte
     << "M, kernel output used size:"
     << mem_buf_used_stat[static_cast<int>(memory::mem_pool::MemType::kKernel)] / kMBToByte
     << "M, other used size:" << other_used_size / kMBToByte << "M.\n";
  return ss.str();
}

const std::pair<size_t, size_t> AbstractDynamicMemPool::FreeIdleMemsByEagerFree() {
  if (!IsEnableVmm() && !IsEnableEagerFree()) {
    LOG_OUT << "FreeIdleMemsByEagerFree is not allowed since vmm is not enabled.";
    return std::make_pair(0L, 0L);
  }

  LOG_OUT << "Free idle mems by eager free start, allocator size : " << streamIdAllocators_.size() << ".";
  eagerFreeCount_++;

  size_t total_eager_free_size = 0;
  size_t total_real_free_size = 0;
  for (auto &stream_id_allocator : streamIdAllocators_) {
    const auto [eager_free_size, real_free_size] = stream_id_allocator.second->FreeIdleMemsByEagerFree();
    total_eager_free_size += eager_free_size;
    total_real_free_size += real_free_size;
  }

  size_t not_free_size =
    total_eager_free_size > total_real_free_size ? (total_eager_free_size - total_real_free_size) : 0;
  if (total_real_free_size >= kGBToByte) {
    LOG_OUT << "Eager free count : " << eagerFreeCount_ << ", free memory : " << total_eager_free_size
            << ", real free : " << total_real_free_size << ", not free : " << not_free_size << ".";
  } else {
    LOG_OUT << "Eager free count : " << eagerFreeCount_ << ", free memory : " << total_eager_free_size
            << ", real free : " << total_real_free_size << ", not free : " << not_free_size << ".";
  }

  memStat_.eagerFreeSize_ += total_eager_free_size;
  return {total_eager_free_size, total_real_free_size};
}

size_t AbstractDynamicMemPool::ReleaseFreeBlocks() {
  LOG_OUT << "Release free blocks start.";
  size_t release_size = 0;
  for (auto &stream_id_allocator : streamIdAllocators_) {
    release_size += stream_id_allocator.second->ReleaseFreeBlocks();
  }
  LOG_OUT << "Release free blocks size : " << release_size << ".";
  return release_size;
}

size_t AbstractDynamicMemPool::ReleaseCustomFreeBlocks() {
  LOG_OUT << "Release custom free blocks start.";
  size_t release_size = 0;
  for (auto &customized_allocator : customizedAllocators_) {
    release_size += customized_allocator.second->ReleaseFreeBlocks();
  }
  LOG_OUT << "Release custom free blocks size : " << release_size << ".";
  return release_size;
}

// The statistics information.
size_t AbstractDynamicMemPool::TotalMemStatistics() const {
  if (IsEnableVmm()) {
    return GetVmmUsedMemSize() + memStat_.customAllocSize_;
  }
  return memStat_.allocSize_ + memStat_.customAllocSize_;
}

size_t AbstractDynamicMemPool::TotalUsedMemStatistics() const { return memStat_.usedSize_; }

size_t AbstractDynamicMemPool::TotalUsedByEventMemStatistics() const { return memStat_.usedByEventSize_; }

size_t AbstractDynamicMemPool::TotalIdleMemStatistics() const { return memStat_.IdleSize(); }

size_t AbstractDynamicMemPool::TotalEagerFreeMemStatistics() const { return memStat_.eagerFreeSize_; }

size_t AbstractDynamicMemPool::UsedMemPeakStatistics() const { return memStat_.peakSize_; }

size_t AbstractDynamicMemPool::MaxMemAllocatedStatistics() const { return memStat_.iterUsedPeakSize_; }

size_t AbstractDynamicMemPool::MaxMemReservedStatistics() const { return memStat_.iterAllocPeakSize_; }

size_t AbstractDynamicMemPool::ActualPeakStatistics() const {
  if (IsEnableVmm()) {
    return GetVmmUsedMemSize() + memStat_.customAllocSize_;
  }

  size_t peak_size = 0;
  for (auto &stream_id_allocator : streamIdAllocators_) {
    peak_size += stream_id_allocator.second->ActualPeakSize();
  }
  for (auto &customized_allocator : customizedAllocators_) {
    peak_size += customized_allocator.second->ActualPeakSize();
  }
  return peak_size;
}

std::unordered_map<std::string, std::size_t> AbstractDynamicMemPool::BlockCountsStatistics() const {
  LockGuard lock(lock_);
  size_t persistent_block_count = 0;
  size_t common_block_count = 0;
  for (const auto &[allocator_info, allocator_ptr] : streamIdAllocators_) {
    if (allocator_info.from_persistent_mem) {
      persistent_block_count += allocator_ptr->memBlocks_.size();
    } else {
      common_block_count += allocator_ptr->memBlocks_.size();
    }
  }
  std::unordered_map<std::string, size_t> block_counts;
  block_counts[kPersistentMemPoolType] = persistent_block_count;
  block_counts[kCommonMemPoolType] = common_block_count;
  return block_counts;
}

std::unordered_map<std::string, std::size_t> AbstractDynamicMemPool::BlockUnitSizeStatistics() const {
  LockGuard lock(lock_);
  std::unordered_map<std::string, size_t> block_units;
  block_units[kPersistentMemPoolType] = persistUnitSize_;
  block_units[kCommonMemPoolType] = commonUnitSize_;
  return block_units;
}

std::unordered_map<device::DeviceMemPtr, std::unordered_map<std::string, size_t>>
AbstractDynamicMemPool::CommonMemBlocksInfoStatistics() const {
  LockGuard lock(lock_);
  std::unordered_map<device::DeviceMemPtr, std::unordered_map<std::string, size_t>> block_infos;
  for (const auto &[allocator_info, allocator_ptr] : streamIdAllocators_) {
    if (!allocator_info.from_persistent_mem) {
      const auto &mem_blocks = allocator_ptr->memBlocks_;
      for (const auto mem_block : mem_blocks) {
        std::unordered_map<std::string, size_t> block_info;
        block_info[kBlockMemorySize] = mem_block->size_;
        block_info[kBlockStreamId] = mem_block->streamId_;
        block_infos[(std::string *)(mem_block->addr_)] = block_info;
      }
    }
  }
  return block_infos;
}

std::unordered_map<device::DeviceMemPtr, std::unordered_map<std::string, size_t>>
AbstractDynamicMemPool::PersistentMemBlocksInfoStatistics() const {
  LockGuard lock(lock_);
  std::unordered_map<device::DeviceMemPtr, std::unordered_map<std::string, size_t>> block_infos;
  for (const auto &[allocator_info, allocator_ptr] : streamIdAllocators_) {
    if (allocator_info.from_persistent_mem) {
      const auto &mem_blocks = allocator_ptr->memBlocks_;
      for (const auto mem_block : mem_blocks) {
        std::unordered_map<std::string, size_t> block_info;
        block_info[kBlockMemorySize] = mem_block->size_;
        block_info[kBlockStreamId] = mem_block->streamId_;
        block_infos[(std::string *)(mem_block->addr_)] = block_info;
      }
    }
  }
  return block_infos;
}

void AbstractDynamicMemPool::ResetMaxMemReserved() {
  LockGuard lock(lock_);
  memStat_.iterAllocPeakSize_ = IsEnableVmm() ? GetVmmUsedMemSize() + memStat_.customAllocSize_
                                                  : memStat_.allocSize_ + memStat_.customAllocSize_;
}

void AbstractDynamicMemPool::ResetMaxMemAllocated() {
  LockGuard lock(lock_);
  memStat_.iterUsedPeakSize_ = memStat_.usedSize_;
}

AbstractEnhancedDynamicMemPool::AbstractEnhancedDynamicMemPool() {}

void AbstractEnhancedDynamicMemPool::ReportMemoryPoolInfo() {
  // Report memory data to profiler.
  if (memoryProfilerCallback_) {
    memoryProfilerCallback_();
  }
}

void AbstractEnhancedDynamicMemPool::ReportMemoryPoolMallocInfoToMstx(void *addr, size_t size) {
  if (memoryMallocMstxCallback_) {
    memoryMallocMstxCallback_(addr, size);
  }
}

void AbstractEnhancedDynamicMemPool::ReportMemoryPoolFreeInfoToMstx(void *addr) {
  if (memoryFreeMstxCallback_) {
    memoryFreeMstxCallback_(addr);
  }
}

MemoryTimeEventPtr AbstractEnhancedDynamicMemPool::GenAllocateMemoryTimeEvent(const void *addr, size_t size,
                                                                              uint32_t stream_id, bool from_persistent,
                                                                              bool is_persistent) {
  auto time_event = std::make_shared<MemoryTimeEvent>();
  time_event->createdAt_ = static_cast<uint64_t>(
    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now().time_since_epoch())
      .count());
  time_event->addr_ = const_cast<void *>(addr);
  time_event->size_ = size;
  time_event->fromPersistent_ = static_cast<uint8_t>(from_persistent);
  time_event->isPersistent_ = static_cast<uint8_t>(is_persistent);
  time_event->streamId_ = stream_id;
  time_event->runMode_ = DynamicMemAllocatorDebugInfo::GetDebugInfo().runMode_;
  time_event->usedSize_ = memStat_.usedSize_;
  time_event->peakSize_ = memStat_.peakSize_;
  time_event->allocSize_ = TotalMemStatistics();
  time_event->usedByEventSize_ = memStat_.usedByEventSize_;
  time_event->eagerFreeSize_ = memStat_.eagerFreeSize_;
  time_event->owner_ = DynamicMemAllocatorDebugInfo::GetDebugInfo().name_;
  time_event->allocType_ = static_cast<uint8_t>(DynamicMemAllocatorDebugInfo::GetDebugInfo().type_);
  return time_event;
}

MemoryTimeEventPtr AbstractEnhancedDynamicMemPool::GenFreeMemoryTimeEvent(const void *addr) {
  auto time_event = std::make_shared<MemoryTimeEvent>();
  time_event->createdAt_ = static_cast<uint64_t>(
    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now().time_since_epoch())
      .count());
  time_event->addr_ = const_cast<void *>(addr);
  const size_t time_event_free_size = -1;
  time_event->size_ = time_event_free_size;
  time_event->usedSize_ = memStat_.usedSize_;
  time_event->peakSize_ = memStat_.peakSize_;
  time_event->allocSize_ = TotalMemStatistics();
  time_event->usedByEventSize_ = memStat_.usedByEventSize_;
  time_event->eagerFreeSize_ = memStat_.eagerFreeSize_;
  return time_event;
}
}  // namespace device
}  // namespace mrt
