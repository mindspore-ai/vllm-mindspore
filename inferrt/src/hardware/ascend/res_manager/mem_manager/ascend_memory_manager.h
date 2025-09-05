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

#ifndef INFERRT_SRC_HARDWARE_ASCEND_ASCEND_MEMORY_MANAGER_H_
#define INFERRT_SRC_HARDWARE_ASCEND_ASCEND_MEMORY_MANAGER_H_

#include <vector>
#include <string>
#include <chrono>

#include <unordered_map>
#include "hardware/hardware_abstract/memory_manager.h"
#include "hardware/ascend/res_manager/mem_manager/ascend_memory_pool.h"

namespace mrt {
namespace device {
namespace ascend {
class MRT_EXPORT AscendMemoryManager : public MemoryManager {
 public:
  AscendMemoryManager() = default;
  ~AscendMemoryManager() override = default;

  void Initialize() override;
  void Finalize() override;
  void ResetDynamicMemory() override;
  void ClearGlobalIdleMem() override;
  void *MallocMemFromMemPool(size_t size, bool fromPersistentMem, bool needRecycle = false,
                             uint32_t streamId = kDefaultStreamIndex) override;
  void FreeMemFromMemPool(void *devicePtr) override;
  size_t GetMaxUsedMemorySize() const override;
  uint64_t GetMsMaxMemSize() const;
  std::vector<void *> MallocContinuousMemFromMemPool(const std::vector<size_t> &sizeList,
                                                     uint32_t streamId = kDefaultStreamIndex) override {
    return AscendMemoryPool::GetInstance().AllocContinuousTensorMem(sizeList, streamId);
  }

  size_t GetAvailableMemSize() override;
  uint64_t GetMsUsedHbmSize() const;

  // Relevant function to manage memory statistics
  size_t GetTotalMemStatistics() const override;
  size_t GetTotalUsedMemStatistics() const override;
  size_t GetTotalIdleMemStatistics() const override;
  size_t GetTotalEagerFreeMemStatistics() const override;
  size_t GetUsedMemPeakStatistics() const override;
  size_t GetReservedMemPeakStatistics() const override;
  std::unordered_map<std::string, std::size_t> GetBlockCountsStatistics() const override;
  std::unordered_map<std::string, std::size_t> GetBlockUnitSizeStatistics() const override;
  std::unordered_map<device::DeviceMemPtr, std::unordered_map<std::string, size_t>> GetCommonMemBlocksInfoStatistics()
    const override;
  std::unordered_map<device::DeviceMemPtr, std::unordered_map<std::string, size_t>>
  GetPersistentMemBlocksInfoStatistics() const override;
  void ResetMaxMemoryReserved() override;
  void ResetMaxMemoryAllocated() override;
  size_t EmptyCache() override;

  DynamicMemPool *GetMemoryPool() override;

 protected:
  uint8_t *MallocStaticMem(size_t size, bool communicationMem, uint32_t graphId) override;
  uint8_t *MallocDynamicMem(size_t size, bool communicationMem) override;
};

class MRT_EXPORT EnhancedAscendMemoryManager : public AscendMemoryManager {
 public:
  EnhancedAscendMemoryManager() = default;
  ~EnhancedAscendMemoryManager() override = default;

  void Initialize() override;

  void Finalize() override;

  void *MallocMemFromMemPool(size_t size, bool fromPersistentMem, bool needRecycle, uint32_t streamId) override;

 private:
  inline uint64_t GetCurrentTick() {
    auto &&ts = std::chrono::system_clock::now();
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(ts.time_since_epoch()).count());
  }

  std::vector<size_t> allocCosts_;
};
}  // namespace ascend
}  // namespace device
}  // namespace mrt
#endif  // INFERRT_SRC_HARDWARE_ASCEND_ASCEND_MEMORY_MANAGER_H_
