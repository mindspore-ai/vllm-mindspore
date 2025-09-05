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

#include "hardware/hardware_abstract/memory_manager.h"
#include <string>
#include "common/common.h"

namespace mrt {
namespace device {
constexpr size_t kAlignBytes = 32;

size_t MemoryManager::GetCommonAlignSize(size_t inputSize) {
  return ((inputSize + kMemAlignSize + kAlignBytes - 1) / kMemAlignSize) * kMemAlignSize;
}

size_t MemoryManager::GetCommunicationAlignSize(size_t inputSize) {
  return ((inputSize + kMemAlignSize - 1) / kMemAlignSize) * kMemAlignSize + kTwiceMemAlignSize;
}

void MemoryManager::FreeMemFromMemPool(void *devicePtr) {
  if (devicePtr == nullptr) {
    LOG_ERROR << "FreeMemFromMemPool devicePtr is null.";
  }
}

uint8_t *MemoryManager::MallocWorkSpaceMem(size_t size) { return MallocDynamicMem(size, false); }

uint8_t *MemoryManager::MallocDynamicMem(size_t size, bool communicationMem) {
  LOG_OUT << "Call default dynamic malloc " << size << " v " << communicationMem;
  return nullptr;
}

void *MemoryManager::MallocMemFromMemPool(size_t size, bool fromPersistentMem, bool, uint32_t streamId) {
  if (size == 0) {
    LOG_ERROR << "MallocMemFromMemPool size is 0.";
  }
  return nullptr;
}

std::vector<void *> MemoryManager::MallocContinuousMemFromMemPool(const std::vector<size_t> &sizeList,
                                                                  uint32_t streamId) {
  if (sizeList.empty()) {
    LOG_ERROR << "MallocContinuousMemFromMemPool size list's size is 0.";
  }
  std::vector<void *> devicePtrList;
  for (size_t i = 0; i < sizeList.size(); ++i) {
    (void)devicePtrList.emplace_back(nullptr);
  }
  return devicePtrList;
}
}  // namespace device
}  // namespace mrt
