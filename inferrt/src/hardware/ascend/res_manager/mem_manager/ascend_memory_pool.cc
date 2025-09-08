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

#include "hardware/ascend/res_manager/mem_manager/ascend_memory_pool.h"

#include <fcntl.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <atomic>
#include <set>

#include "common/common.h"
#include "hardware/ascend/res_manager/mem_manager/ascend_vmm_adapter.h"
#include "hardware/ascend/res_manager/ascend_stream_manager.h"

namespace mrt {
namespace device {
namespace ascend {
constexpr size_t kByteOffset = 8;
constexpr uint32_t kDefaultDispatchThreadsNum = 5;
constexpr uint32_t kDefaultOpThreadsNum = 25;
constexpr float kDefaultMemInitSize = 2.0;
constexpr float kDefaultMemBlockIncreaseSize = 1.0;
constexpr float kDefaultMemMaxSize = 1024.0;

DefaultAscendMemoryPool::DefaultAscendMemoryPool() {
  LOG_OUT << "DefaultAscendMemoryPool constructed.";
  SetEnableVmm(AscendVmmAdapter::GetInstance().IsEnabled());
}

size_t DefaultAscendMemoryPool::EmptyCache() {
  LockGuard lock(AbstractDynamicMemPool::lock());
  AbstractEnhancedDynamicMemPool::WaitPipelineHelper();
  AbstractAscendMemoryPoolSupport::SyncAllStreams();
  size_t release_free_size = 0;
  if (MS_UNLIKELY(!customizedAllocators_.empty())) {
    release_free_size += ReleaseCustomFreeBlocks();
  }
  if (IsEnableVmm()) {
    AbstractEnhancedDynamicMemPool::FreeIdleMemsByEagerFree();
    release_free_size += AbstractAscendMemoryPoolSupport::EmptyCache();
    return release_free_size;
  } else if (IsEnableEagerFree()) {
    auto ret = AbstractEnhancedDynamicMemPool::FreeIdleMemsByEagerFree();
    LOG_OUT << "Eager free memory size is " << ret.second << ".";
    release_free_size += ret.second;
    return release_free_size;
  }

  LOG_OUT << "Vmm is not enabled, try to release free blocks.";
  // // disable ge kernel use two pointer mem adapter, not support free.
  // if (IsDisableGeKernel()) {
  //   return 0L;
  // }
  release_free_size += ReleaseFreeBlocks();
  return release_free_size;
}

void DefaultAscendMemoryPool::EnablePluggableAllocator(std::function<MallocFuncType> alloc_fn,
                                                       std::function<FreeFuncType> free_fn) {
  customAllocFn_ = alloc_fn;
  customFreeFn_ = free_fn;
  enableCustomAllocator_ = true;
}

void DefaultAscendMemoryPool::DisablePluggableAllocator() {
  enableCustomAllocator_ = false;
  return;
}

DefaultEnhancedAscendMemoryPool::DefaultEnhancedAscendMemoryPool(const DefaultAscendMemoryPoolPtr &instance)
    : instance_(instance) {
  LOG_OUT << "DefaultEnhancedAscendMemoryPool constructed.";
  instance_->SetEnableVmm(AscendVmmAdapter::GetInstance().IsEnabled());
}

void DefaultEnhancedAscendMemoryPool::ReleaseDeviceRes() {
  LOG_OUT << "Start release device res.";
  instance_->ReleaseDeviceRes();
}

DeviceMemPtr DefaultEnhancedAscendMemoryPool::AllocTensorMem(size_t size, bool from_persistent_mem, bool need_recycle,
                                                             uint32_t stream_id) {
  size_t align_size = AlignMemorySize(size);
  LOG_OUT << "Allocate tensor mem, size : " << size << ", align_size : " << align_size
          << ", need_recycle : " << need_recycle << ".";
  LockGuard lock(instance_->lock());
  const auto [mem_buf, allocator] = instance_->AllocMemBuf(align_size, from_persistent_mem, stream_id);
  if (mem_buf == nullptr) {
    LOG_OUT << "Allocate tensor mem, return nullptr.";
    // Dump mem pool state info and debug info when alloc tensor failed.
    DumpDynamicMemPoolStateInfo();
    DumpDynamicMemPoolDebugInfo();
    return nullptr;
  }

  mem_buf->SetDebugInfo();
  instance_->addr_mem_buf_allocators().emplace(mem_buf->addr_, std::make_pair(mem_buf, allocator));
  auto device_addr = mem_buf->addr_;

  instance_->ReportMemoryPoolInfo();
  instance_->ReportMemoryPoolMallocInfoToMstx(device_addr, align_size);

  LOG_OUT << "Allocate tensor mem, return : " << mem_buf->ToJson() << ", stat info : " << instance_->mem_stat().ToJson()
          << ".";
  return device_addr;
}

std::vector<DeviceMemPtr> DefaultEnhancedAscendMemoryPool::AllocContinuousTensorMem(
  const std::vector<size_t> &size_list, uint32_t stream_id) {
  LOG_OUT << "Alloc continuous tensor mem, stream id : " << stream_id << ".";
  const auto &continuous_addrs = instance_->AllocContinuousTensorMem(size_list, stream_id);
  if (continuous_addrs.size() != size_list.size()) {
    return continuous_addrs;
  }
  if (continuous_addrs.size() == 1 && continuous_addrs[0] == nullptr) {
    return continuous_addrs;
  }
  return continuous_addrs;
}

void DefaultEnhancedAscendMemoryPool::FreeTensorMem(const DeviceMemPtr &device_addr) {
  LOG_OUT << "Free tensor mem, device addr : " << device_addr << ".";
  LockGuard lock(instance_->lock());
  DoFreeTensorMem(device_addr);
}

bool DefaultEnhancedAscendMemoryPool::DoFreeTensorMem(const DeviceMemPtr &device_addr) {
  void *enhanced_device_addr = device_addr;
  bool ret = instance_->DoFreeTensorMem(device_addr);
  LOG_OUT << "Do free tensor mem : " << enhanced_device_addr << ", return : " << ret << ".";
  return ret;
}

void DefaultEnhancedAscendMemoryPool::FreePartTensorMems(const std::vector<DeviceMemPtr> &free_addrs,
                                                         const std::vector<DeviceMemPtr> &keep_addrs,
                                                         const std::vector<size_t> &keep_addr_sizes) {
  LOG_OUT << "Free part tensor mems.";
  LockGuard lock(instance_->lock());

  const auto keep_mem_bufs = instance_->DoFreePartTensorMems(free_addrs, keep_addrs, keep_addr_sizes);
}

void DefaultEnhancedAscendMemoryPool::DefragMemory() {
  if (lastVmmUsedSize_ == 0) {
    lastVmmUsedSize_ = GetVmmUsedMemSize();
  } else {
    size_t vmm_used_size = GetVmmUsedMemSize();
    if (vmm_used_size > lastVmmUsedSize_) {
      LOG_OUT << "Current vmm used size : " << vmm_used_size
              << " is bigger than last vmm used size : " << lastVmmUsedSize_ << ".";
      lastVmmUsedSize_ = vmm_used_size;
    }
  }

  instance_->DefragMemory();
}

void DefaultEnhancedAscendMemoryPool::DumpDynamicMemPoolStateInfo() { instance_->DumpDynamicMemPoolStateInfo(); }

const std::pair<size_t, size_t> DefaultEnhancedAscendMemoryPool::FreeIdleMemsByEagerFree() {
  const auto [eager_free_size, real_free_size] = instance_->FreeIdleMemsByEagerFree();
  return {eager_free_size, real_free_size};
}

bool DefaultEnhancedAscendMemoryPool::WaitEvent(int64_t task_id_on_stream, uint32_t user_stream_id,
                                                uint32_t memory_stream_id) {
  LockGuard lock(instance_->lock());
  auto key = std::make_pair(user_stream_id, memory_stream_id);
  auto iter = instance_->stream_pair_mem_bufs().find(key);
  if (iter == instance_->stream_pair_mem_bufs().end()) {
    return false;
  }

  auto mem_bufs_ = iter->second;
  for (const auto &mem_buf : mem_bufs_) {
    LOG_OUT << "Wait event for : " << mem_buf->ToJson() << ".";
    mem_buf->WaitEvent(task_id_on_stream, user_stream_id);
    // Remove event and try to free memory.
    if (mem_buf->IsEventNotUsed()) {
      instance_->mem_stat().usedByEventSize_ -= mem_buf->size_;
      // Force clear all mem bufs.
      for (auto &stream_pair_mem_bufs : instance_->stream_pair_mem_bufs()) {
        (void)stream_pair_mem_bufs.second.erase(mem_buf);
      }
      if (mem_buf->status_ == DynamicMemBufStatus::kMemBufUsedByEvent) {
        (void)DoFreeTensorMem(mem_buf->addr_);
      }
    }
  }
  return true;
}

bool DefaultEnhancedAscendMemoryPool::WaitEvent(int64_t task_id_on_stream, uint32_t memory_stream_id) {
  LockGuard lock(instance_->lock());
  for (auto &stream_pair_mem_bufs : instance_->stream_pair_mem_bufs()) {
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
        instance_->mem_stat().usedByEventSize_ -= mem_buf->size_;
        // Force clear all mem bufs.
        for (auto &kv : instance_->stream_pair_mem_bufs()) {
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

bool DefaultEnhancedAscendMemoryPool::SyncAllEvents() {
  LockGuard lock(instance_->lock());
  if (stream_pair_mem_bufs().empty()) {
    return false;
  }

  std::set<MemBuf *> carry_event_mem_bufs;
  for (const auto &stream_pair_mem_buf : instance_->stream_pair_mem_bufs()) {
    for (const auto &mem_buf : stream_pair_mem_buf.second) {
      (void)carry_event_mem_bufs.emplace(mem_buf);
    }
  }
  for (auto &mem_buf : carry_event_mem_bufs) {
    if (mem_buf->SyncAllEvents() && mem_buf->status_ == DynamicMemBufStatus::kMemBufUsedByEvent) {
      (void)DoFreeTensorMem(mem_buf->addr_);
    }
  }

  instance_->stream_pair_mem_bufs().clear();
  return true;
}

void DefaultEnhancedAscendMemoryPool::SetRankIdGetter(const std::function<size_t()> &rank_id_getter) {
  instance_->SetRankIdGetter(rank_id_getter);
  if (rank_id_getter != nullptr) {
    rankIdGetter_ = rank_id_getter;
  }
}

BestFitAscendMemoryPool::BestFitAscendMemoryPool() {
  LOG_OUT << "BestFitAscendMemoryPool constructed, older memory allocator is enabled.";
  SetEnableVmm(AscendVmmAdapter::GetInstance().IsEnabled());
}

size_t BestFitAscendMemoryPool::EmptyCache() {
  LOG_OUT << "Best fit memory pool is not supported empty cache.";
  return 0L;
}

// Initialize static member in AscendMemoryPool.
AbstractAscendMemoryPoolSupportPtr AscendMemoryPool::pool_ = nullptr;

AbstractAscendMemoryPoolSupportPtr AscendMemoryPool::instance_ = nullptr;

AbstractAscendMemoryPoolSupportPtr AscendMemoryPool::enhancedInstance_ = nullptr;

AbstractAscendMemoryPoolSupport &AscendMemoryPool::GetInstance() {
  static std::once_flag flag;
  std::call_once(flag, [&]() {
    if (UseOldMemoryPool()) {
      instance_ = std::make_shared<BestFitAscendMemoryPool>();
      enhancedInstance_ = instance_;
    } else {
      const auto &memory_pool = std::make_shared<DefaultAscendMemoryPool>();
      instance_ = memory_pool;
      enhancedInstance_ = std::make_shared<DefaultEnhancedAscendMemoryPool>(memory_pool);
    }
    // Initialize instance and set ptr.
    float init_size = kDefaultMemInitSize;
    size_t init_size_byte = FloatToSize(init_size * kGBToByte);
    float increase_size = kDefaultMemBlockIncreaseSize;
    size_t increase_size_byte = FloatToSize(increase_size * kGBToByte);
    float max_size = kDefaultMemMaxSize;
    size_t max_size_byte = FloatToSize(max_size * kGBToByte);
    instance_->Initialize(init_size_byte, increase_size_byte, max_size_byte);
    // Set memory mstx callback func.
    if (!UseEnhancedMemoryPool()) {
      pool_ = instance_;
    } else {
      pool_ = enhancedInstance_;
    }
  });
  return *pool_;
}

void AscendMemoryPool::SetEnhancedMemoryPool(bool enable) {
  LOG_OUT << "Set enhanced memory pool : " << enable << ".";
  if (enable) {
    pool_ = enhancedInstance_;
  } else {
    pool_ = instance_;
  }
}

bool AscendMemoryPool::UseOldMemoryPool() {
  return false;
  // if (memory::mem_pool::IsDisableAllocConfig(memory::mem_pool::kAllocMemoryPool)) {
  //   return false;
  // }
  // return IsDisableGeKernel() || memory::mem_pool::IsEnableAllocConfig(memory::mem_pool::kAllocMemoryPool);
}

// Use enhanced memory pool when enable debug, enable log, enable prof, dry run and so on.
bool AscendMemoryPool::UseEnhancedMemoryPool() { return false; }
}  // namespace ascend
}  // namespace device
}  // namespace mrt
