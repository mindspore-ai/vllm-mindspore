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
#include "hardware/ascend/res_manager/mem_manager/ascend_memory_manager.h"

#include <algorithm>
#include <string>
#include <chrono>
#include <numeric>
#include <unordered_map>

#include "hardware/ascend/res_manager/mem_manager/ascend_memory_adapter.h"
#include "hardware/ascend/res_manager/ascend_stream_manager.h"
#include "hardware/ascend/res_manager/symbol_interface/acl_rt_symbol.h"
#include "hardware/ascend/res_manager/symbol_interface/symbol_utils.h"
#include "common/common.h"

namespace mrt {
namespace device {
namespace ascend {
void AscendMemoryManager::Initialize() { (void)AscendMemAdapter::GetInstance()->Initialize(); }

void AscendMemoryManager::Finalize() {
  AscendMemoryPool::GetInstance().ReleaseDeviceRes();
  (void)AscendMemAdapter::GetInstance()->DeInitialize();
}

void AscendMemoryManager::ResetDynamicMemory() { AscendMemAdapter::GetInstance()->ResetDynamicMemory(); }

void AscendMemoryManager::ClearGlobalIdleMem() { AscendMemoryPool::GetInstance().ResetIdleMemBuf(); }

uint64_t AscendMemoryManager::GetMsMaxMemSize() const { return AscendMemAdapter::GetInstance()->MaxHbmSizeForMs(); }

uint64_t AscendMemoryManager::GetMsUsedHbmSize() const { return AscendMemAdapter::GetInstance()->GetMsUsedHbmSize(); }

void *AscendMemoryManager::MallocMemFromMemPool(size_t size, bool fromPersistentMem, bool needRecycle,
                                                uint32_t streamId) {
  auto alignSize = GetCommonAlignSize(size);
  return AscendMemoryPool::GetInstance().AllocTensorMem(alignSize, fromPersistentMem, needRecycle, streamId);
}

void AscendMemoryManager::FreeMemFromMemPool(void *devicePtr) {
  AscendMemoryPool::GetInstance().FreeTensorMem(devicePtr);
}

size_t AscendMemoryManager::GetMaxUsedMemorySize() const { return AscendMemoryPool::GetInstance().GetMaxUsedMemSize(); }

// Relevant function to manage memory statistics
size_t AscendMemoryManager::GetTotalMemStatistics() const {
  return AscendMemoryPool::GetInstance().TotalMemStatistics();
}
size_t AscendMemoryManager::GetTotalUsedMemStatistics() const {
  return AscendMemoryPool::GetInstance().TotalUsedMemStatistics();
}
size_t AscendMemoryManager::GetTotalIdleMemStatistics() const {
  return AscendMemoryPool::GetInstance().TotalIdleMemStatistics();
}
size_t AscendMemoryManager::GetTotalEagerFreeMemStatistics() const {
  return AscendMemoryPool::GetInstance().TotalEagerFreeMemStatistics();
}
size_t AscendMemoryManager::GetUsedMemPeakStatistics() const {
  return AscendMemoryPool::GetInstance().MaxMemAllocatedStatistics();
}
size_t AscendMemoryManager::GetReservedMemPeakStatistics() const {
  return AscendMemoryPool::GetInstance().MaxMemReservedStatistics();
}
std::unordered_map<std::string, std::size_t> AscendMemoryManager::GetBlockCountsStatistics() const {
  return AscendMemoryPool::GetInstance().BlockCountsStatistics();
}
std::unordered_map<std::string, std::size_t> AscendMemoryManager::GetBlockUnitSizeStatistics() const {
  return AscendMemoryPool::GetInstance().BlockUnitSizeStatistics();
}
std::unordered_map<device::DeviceMemPtr, std::unordered_map<std::string, size_t>>
AscendMemoryManager::GetCommonMemBlocksInfoStatistics() const {
  return AscendMemoryPool::GetInstance().CommonMemBlocksInfoStatistics();
}
std::unordered_map<device::DeviceMemPtr, std::unordered_map<std::string, size_t>>
AscendMemoryManager::GetPersistentMemBlocksInfoStatistics() const {
  return AscendMemoryPool::GetInstance().PersistentMemBlocksInfoStatistics();
}
void AscendMemoryManager::ResetMaxMemoryReserved() { AscendMemoryPool::GetInstance().ResetMaxMemReserved(); }
void AscendMemoryManager::ResetMaxMemoryAllocated() { AscendMemoryPool::GetInstance().ResetMaxMemAllocated(); }
size_t AscendMemoryManager::EmptyCache() { return AscendMemoryPool::GetInstance().EmptyCache(); }

uint8_t *AscendMemoryManager::MallocStaticMem(size_t size, bool communicationMem, uint32_t graphId) {
  size_t alignSize = 0;
  if (communicationMem) {
    alignSize = GetCommunicationAlignSize(size);
  } else {
    alignSize = GetCommonAlignSize(size);
  }
  LOG_OUT << "Malloc Memory for Static: size[" << alignSize << "] communicationMem:" << communicationMem;

  uint8_t *allocAddress = reinterpret_cast<uint8_t *>(AscendMemoryPool::GetInstance().AllocTensorMem(alignSize));
  if (allocAddress != nullptr) {
    // create protect area [kMemAlignSize -- data -- kMemAlignSize] for communication node memory
    return communicationMem ? allocAddress + kMemAlignSize : allocAddress;
  }
  LOG_ERROR << "#umsg#Framework Error Message:#umsg#Fail to alloc memory, size: " << alignSize
            << "B, memory statistics:" << AscendMemAdapter::GetInstance()->DevMemStatistics();
  return 0;
}

uint8_t *AscendMemoryManager::MallocDynamicMem(size_t size, bool communicationMem) {
  size_t alignSize = 0;
  if (communicationMem) {
    alignSize = GetCommunicationAlignSize(size);
  } else {
    alignSize = GetCommonAlignSize(size);
  }
  LOG_OUT << "Malloc Memory for Dynamic: size[" << alignSize << "] communicationMem: " << communicationMem;

  uint8_t *allocAddress = reinterpret_cast<uint8_t *>(AscendMemAdapter::GetInstance()->MallocDynamicDevMem(alignSize));
  CHECK_IF_NULL(allocAddress);
  // create protect area [kMemAlignSize -- data -- kMemAlignSize] for communication node memory
  return communicationMem ? allocAddress + kMemAlignSize : allocAddress;
}

size_t AscendMemoryManager::GetAvailableMemSize() {
  auto availableMemSize = AscendMemoryPool::GetInstance().free_mem_size() +
                          AscendMemoryPool::GetInstance().TotalMemStatistics() -
                          AscendMemoryPool::GetInstance().TotalUsedMemStatistics();
  return availableMemSize;
}

DynamicMemPool *AscendMemoryManager::GetMemoryPool() {
  if (MS_UNLIKELY(memoryPool_ == nullptr)) {
    memoryPool_ = &(AscendMemoryPool::GetInstance());
  }
  return memoryPool_;
}

void EnhancedAscendMemoryManager::Initialize() {
  AscendMemoryManager::Initialize();
  LOG_OUT << "EnhancedAscendMemoryManager initialize.";
  allocCosts_.clear();
}

void EnhancedAscendMemoryManager::Finalize() {
  AscendMemoryManager::Finalize();
  LOG_OUT << "EnhancedAscendMemoryManager finalize";
  std::sort(allocCosts_.begin(), allocCosts_.end());
  // Calculate mean and median, then print them.
  auto totalSize = allocCosts_.size();
  if (totalSize == 0) {
    LOG_OUT << "No memory operation.";
    return;
  }
  double median = 0;
  if (totalSize & 1) {
    median = (allocCosts_[totalSize >> 1] + allocCosts_[(totalSize >> 1) + 1]) >> 1;
  } else {
    median = allocCosts_[totalSize >> 1];
  }
  LOG_OUT << "EnhancedAscendMemoryManager median : " << median << "ns.";

  double sum = std::accumulate(allocCosts_.begin(), allocCosts_.end(), 0.0);
  double mean = sum / totalSize;
  LOG_OUT << "EnhancedAscendMemoryManager mean : " << mean << "ns.";

  const double costHighWater = 1800;
  if (median > costHighWater || mean > costHighWater) {
    LOG_OUT << "EnhancedAscendMemoryManager check failed, median : " << median << ", mean : " << mean;
  }
}

void *EnhancedAscendMemoryManager::MallocMemFromMemPool(size_t size, bool fromPersistentMem, bool needRecycle,
                                                        uint32_t streamId) {
  auto startTick = GetCurrentTick();
  auto ret = AscendMemoryManager::MallocMemFromMemPool(size, fromPersistentMem, needRecycle, streamId);
  auto cost = GetCurrentTick() - startTick;
  (void)allocCosts_.emplace_back(cost);
  LOG_OUT << "Malloc memory cost : " << cost << "ns.";
  return ret;
}
}  // namespace ascend
}  // namespace device
}  // namespace mrt
