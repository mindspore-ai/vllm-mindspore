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

#ifndef INFERRT_SRC_HARDWARE_ASCEND_ASCEND_MEMORY_POOL_H_
#define INFERRT_SRC_HARDWARE_ASCEND_ASCEND_MEMORY_POOL_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "hardware/hardware_abstract/memory/abstract_dynamic_mem_pool.h"
#include "common/visible.h"
#include "hardware/ascend/res_manager/mem_manager/abstract_ascend_memory_pool_support.h"

namespace mrt {
namespace device {
namespace ascend {

class MRT_EXPORT DefaultAscendMemoryPool : public AbstractAscendMemoryPoolSupport,
                                                public AbstractEnhancedDynamicMemPool {
 public:
  DefaultAscendMemoryPool();
  DefaultAscendMemoryPool(const DefaultAscendMemoryPool &) = delete;
  DefaultAscendMemoryPool &operator=(const DefaultAscendMemoryPool &) = delete;
  ~DefaultAscendMemoryPool() override = default;

  std::string GetMemoryPoolType() const override { return "DefaultAscendMemoryPool"; }

  void SetMemPoolBlockSize(size_t available_device_mem_size) override {
    return AbstractAscendMemoryPoolSupport::SetMemPoolBlockSize(available_device_mem_size);
  }

  size_t CalMemBlockAllocSize(size_t size, bool from_persistent_mem, bool need_recycle = false) override {
    return AbstractAscendMemoryPoolSupport::CalMemBlockAllocSize(size, from_persistent_mem, need_recycle);
  }

  const bool IsEnableEagerFree() const override { return AbstractAscendMemoryPoolSupport::IsEnableEagerFree(); }

  size_t EmptyCache() override;

  void EnablePluggableAllocator(std::function<MallocFuncType> alloc_fn, std::function<FreeFuncType> free_fn) override;
  void DisablePluggableAllocator() override;
};
using DefaultAscendMemoryPoolPtr = std::shared_ptr<DefaultAscendMemoryPool>;

class MRT_EXPORT DefaultEnhancedAscendMemoryPool : public DefaultAscendMemoryPool {
 public:
  explicit DefaultEnhancedAscendMemoryPool(const DefaultAscendMemoryPoolPtr &instance);
  DefaultEnhancedAscendMemoryPool(const DefaultEnhancedAscendMemoryPool &) = delete;
  DefaultEnhancedAscendMemoryPool &operator=(const DefaultEnhancedAscendMemoryPool &) = delete;
  ~DefaultEnhancedAscendMemoryPool() override = default;

  // Wrap enhanced function.
  void Initialize(size_t init_size, size_t increase_size, size_t max_size) override {
    instance_->Initialize(init_size, increase_size, max_size);
  }

  void ReleaseDeviceRes() override;

  DeviceMemPtr AllocTensorMem(size_t size, bool from_persistent_mem = false, bool need_recycle = false,
                              uint32_t stream_id = kDefaultStreamIndex) override;

  std::vector<DeviceMemPtr> AllocContinuousTensorMem(const std::vector<size_t> &size_list,
                                                     uint32_t stream_id = kDefaultStreamIndex) override;

  void FreeTensorMem(const DeviceMemPtr &device_addr) override;

  bool DoFreeTensorMem(const DeviceMemPtr &device_addr) override;

  void FreePartTensorMems(const std::vector<DeviceMemPtr> &free_addrs, const std::vector<DeviceMemPtr> &keep_addrs,
                          const std::vector<size_t> &keep_addr_sizes) override;

  std::vector<MemBuf *> DoFreePartTensorMems(const std::vector<DeviceMemPtr> &free_addrs,
                                             const std::vector<DeviceMemPtr> &keep_addrs,
                                             const std::vector<size_t> &keep_addr_sizes) override {
    return instance_->DoFreePartTensorMems(free_addrs, keep_addrs, keep_addr_sizes);
  }

  void DefragMemory() override;

  void DumpDynamicMemPoolStateInfo() override;

  const std::pair<size_t, size_t> FreeIdleMemsByEagerFree() override;

  size_t ReleaseFreeBlocks() override { return instance_->ReleaseFreeBlocks(); }

  // Proxy wrapper for AbstractAscendMemoryPoolSupport
  void ResetIdleMemBuf() const override { instance_->ResetIdleMemBuf(); }

  bool RecordEvent(int64_t task_id_on_stream, uint32_t user_stream_id,
                   const std::vector<std::pair<uint32_t, DeviceMemPtr>> &memory_stream_addresses,
                   const DeviceEventPtr &event) override {
    return instance_->RecordEvent(task_id_on_stream, user_stream_id, memory_stream_addresses, event);
  }

  bool WaitEvent(int64_t task_id_on_stream, uint32_t user_stream_id, uint32_t memory_stream_id) override;

  bool WaitEvent(int64_t task_id_on_stream, uint32_t memory_stream_id) override;

  bool SyncAllEvents() override;

  void EnablePluggableAllocator(std::function<MallocFuncType> alloc_fn, std::function<FreeFuncType> free_fn) override {
    return instance_->EnablePluggableAllocator(alloc_fn, free_fn);
  }

  void DisablePluggableAllocator() override { return instance_->DisablePluggableAllocator(); }

  size_t AlignMemorySize(size_t size) const override { return instance_->AlignMemorySize(size); }

  size_t CalMemBlockAllocSize(size_t size, bool from_persistent_mem, bool need_recycle = false) override {
    return instance_->CalMemBlockAllocSize(size, from_persistent_mem, need_recycle);
  }

  void SetMemPoolBlockSize(size_t available_device_mem_size) override {
    instance_->SetMemPoolBlockSize(available_device_mem_size);
  }

  size_t MemAllocUnitSize(bool from_persistent_mem) const override {
    return instance_->MemAllocUnitSize(from_persistent_mem);
  }

  void SetMemAllocUintSize(size_t common_size, size_t persist_size = kDynamicMemAllocUnitSize) override {
    instance_->SetMemAllocUintSize(common_size, persist_size);
  }

  void *GetMinUsingMemoryAddr() const override { return instance_->GetMinUsingMemoryAddr(); }

  size_t AllocDeviceMem(size_t size, DeviceMemPtr *addr) override { return instance_->AllocDeviceMem(size, addr); }

  bool FreeDeviceMem(const DeviceMemPtr &addr) override { return instance_->FreeDeviceMem(addr); }

  size_t free_mem_size() override { return instance_->free_mem_size(); }

  uint64_t total_mem_size() const override { return instance_->total_mem_size(); }

  size_t GetMaxUsedMemSize() const override { return instance_->GetMaxUsedMemSize(); }

  size_t GetVmmUsedMemSize() const override { return instance_->GetVmmUsedMemSize(); }

  void DumpDynamicMemPoolDebugInfo() override { instance_->DumpDynamicMemPoolDebugInfo(); }

  size_t TotalMemStatistics() const override { return instance_->TotalMemStatistics(); }

  size_t TotalUsedMemStatistics() const override { return instance_->TotalUsedMemStatistics(); }

  size_t TotalUsedByEventMemStatistics() const override { return instance_->TotalUsedByEventMemStatistics(); }

  size_t TotalIdleMemStatistics() const override { return instance_->TotalIdleMemStatistics(); }

  size_t TotalEagerFreeMemStatistics() const override { return instance_->TotalEagerFreeMemStatistics(); }

  size_t UsedMemPeakStatistics() const override { return instance_->UsedMemPeakStatistics(); }

  size_t MaxMemAllocatedStatistics() const override { return instance_->MaxMemAllocatedStatistics(); }

  size_t MaxMemReservedStatistics() const override { return instance_->MaxMemReservedStatistics(); }

  size_t ActualPeakStatistics() const override { return instance_->ActualPeakStatistics(); }

  std::unordered_map<std::string, std::size_t> BlockCountsStatistics() const override {
    return std::move(instance_->BlockCountsStatistics());
  }

  std::unordered_map<std::string, std::size_t> BlockUnitSizeStatistics() const override {
    return std::move(instance_->BlockUnitSizeStatistics());
  }

  std::unordered_map<device::DeviceMemPtr, std::unordered_map<std::string, size_t>> CommonMemBlocksInfoStatistics()
    const override {
    return std::move(instance_->CommonMemBlocksInfoStatistics());
  }

  std::unordered_map<device::DeviceMemPtr, std::unordered_map<std::string, size_t>> PersistentMemBlocksInfoStatistics()
    const override {
    return std::move(instance_->PersistentMemBlocksInfoStatistics());
  }

  void ResetMaxMemReserved() override { instance_->ResetMaxMemReserved(); }

  void ResetMaxMemAllocated() override { instance_->ResetMaxMemAllocated(); }

  const bool IsEnableEagerFree() const override { return instance_->IsEnableEagerFree(); }

  const bool IsEnableVmm() const override { return instance_->IsEnableVmm(); }

  void SetEnableVmm(bool enable_vmm) override { instance_->SetEnableVmm(enable_vmm); }

  const bool SyncAllStreams() override { return instance_->SyncAllStreams(); }

  size_t AllocDeviceMemByEagerFree(size_t size, DeviceMemPtr *addr) override {
    return instance_->AllocDeviceMemByEagerFree(size, addr);
  }

  size_t FreeDeviceMemByEagerFree(const DeviceMemPtr addr, const size_t size) override {
    return instance_->FreeDeviceMemByEagerFree(addr, size);
  }

  size_t MmapDeviceMem(size_t size, DeviceMemPtr addr) override { return instance_->MmapDeviceMem(size, addr); }

  std::string GetMemoryPoolType() const override { return "DefaultEnhancedAscendMemoryPool"; }

  void ReportMemoryPoolInfo() override { instance_->ReportMemoryPoolInfo(); }

  void ReportMemoryPoolMallocInfoToMstx(void *ptr, size_t size) override {
    instance_->ReportMemoryPoolMallocInfoToMstx(ptr, size);
  }

  void ReportMemoryPoolFreeInfoToMstx(void *ptr) override { instance_->ReportMemoryPoolFreeInfoToMstx(ptr); }

  bool IsEnableTimeEvent() override { return instance_->IsEnableTimeEvent(); }

  void SetEnableTimeEvent(bool enable_time_event) override { instance_->SetEnableTimeEvent(enable_time_event); }

  MemoryTimeEventPtr GenAllocateMemoryTimeEvent(const void *addr, size_t size, uint32_t stream_id, bool from_persistent,
                                                bool is_persistent) override {
    return instance_->GenAllocateMemoryTimeEvent(addr, size, stream_id, from_persistent, is_persistent);
  }

  MemoryTimeEventPtr GenFreeMemoryTimeEvent(const void *addr) override {
    return instance_->GenFreeMemoryTimeEvent(addr);
  }

  size_t EmptyCache() override { return instance_->EmptyCache(); }

 protected:
  void SetRankIdGetter(const std::function<size_t()> &rank_id_getter) override;

 private:
  DefaultAscendMemoryPoolPtr instance_;
  size_t lastVmmUsedSize_{0};
};

class MRT_EXPORT BestFitAscendMemoryPool : public AbstractAscendMemoryPoolSupport {
 public:
  BestFitAscendMemoryPool();
  BestFitAscendMemoryPool(const BestFitAscendMemoryPool &) = delete;
  BestFitAscendMemoryPool &operator=(const BestFitAscendMemoryPool &) = delete;
  ~BestFitAscendMemoryPool() override = default;

  void SetMemPoolBlockSize(size_t available_device_mem_size) override {
    return AbstractAscendMemoryPoolSupport::SetMemPoolBlockSize(available_device_mem_size);
  }

  size_t CalMemBlockAllocSize(size_t size, bool from_persistent_mem, bool need_recycle = false) override {
    return AbstractAscendMemoryPoolSupport::CalMemBlockAllocSize(size, from_persistent_mem, need_recycle);
  }

  const bool IsEnableEagerFree() const override { return AbstractAscendMemoryPoolSupport::IsEnableEagerFree(); }

  std::string GetMemoryPoolType() const override { return "BestFitAscendMemoryPool"; }

  size_t EmptyCache() override;
};

class MRT_EXPORT AscendMemoryPool {
 public:
  AscendMemoryPool(const AscendMemoryPool &) = delete;
  AscendMemoryPool &operator=(const AscendMemoryPool &) = delete;

  static AbstractAscendMemoryPoolSupport &GetInstance();

  static void SetEnhancedMemoryPool(bool enable);

 private:
  AscendMemoryPool() {}

  static bool UseOldMemoryPool();

  // Use enhanced memory pool when enable debug, enable log, enable prof, dry run and so on.
  static bool UseEnhancedMemoryPool();

  // Reference to memory pool.
  static AbstractAscendMemoryPoolSupportPtr pool_;

  // Basic memory pool instance with high performance.
  static AbstractAscendMemoryPoolSupportPtr instance_;

  // Memory pool support profiling and debugging.
  static AbstractAscendMemoryPoolSupportPtr enhancedInstance_;
};
}  // namespace ascend
}  // namespace device
}  // namespace mrt

#endif  // INFERRT_SRC_HARDWARE_ASCEND_ASCEND_MEMORY_POOL_H_
