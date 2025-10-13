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
#include "hardware/ascend/res_manager/mem_manager/ascend_vmm_adapter.h"
#include <map>
#include <vector>
#include <tuple>

#include "hardware/ascend/res_manager/symbol_interface/symbol_utils.h"
#include "hardware/ascend/res_manager/symbol_interface/acl_rt_symbol.h"
#include "common/common.h"

#include "hardware/hardware_abstract/collective/collective_manager.h"

namespace mrt {
namespace device {
namespace ascend {
size_t AscendVmmAdapter::GetRoundUpAlignSize(size_t inputSize) const {
  return ((inputSize + vmmAlignSize_ - 1) / vmmAlignSize_) * vmmAlignSize_;
}

size_t AscendVmmAdapter::GetRoundDownAlignSize(size_t inputSize) const {
  return (inputSize / vmmAlignSize_) * vmmAlignSize_;
}

size_t AscendVmmAdapter::GetHandleSize(size_t inputSize) {
  if (inputSize % vmmAlignSize_ != 0) {
    LOG_ERROR << "Input size must be multiple of 2MB, but got " << inputSize;
  }
  return inputSize / vmmAlignSize_;
}

DeviceMemPtr AscendVmmAdapter::FindVmmSegment(const DeviceMemPtr addr) {
  auto it = vmmMap_.upper_bound(addr);
  if (it == vmmMap_.begin()) {
    return nullptr;
  } else {
    --it;
    return it->first;
  }
  return nullptr;
}

void AscendVmmAdapter::ClearAllMemory() {
  for (auto &kv : vmmMap_) {
    if (kv.second == nullptr) {
      continue;
    }
    auto ret = CALL_ASCEND_API(aclrtUnmapMem, kv.first);
    if (ret != ACL_SUCCESS) {
      LOG_ERROR << "Unmap memory failed.";
    }
    ret = CALL_ASCEND_API(aclrtFreePhysical, kv.second);
    if (ret != ACL_SUCCESS) {
      LOG_ERROR << "Free physical memory failed.";
    }
  }
  while (!cachedHandleSets_.empty()) {
    auto handle = *cachedHandleSets_.begin();
    cachedHandleSets_.erase(cachedHandleSets_.begin());
    auto ret = CALL_ASCEND_API(aclrtFreePhysical, handle);
    if (ret != ACL_SUCCESS) {
      LOG_ERROR << "Free physical memory failed.";
    }
  }
  for (auto &addr : allReserveMems_) {
    CALL_ASCEND_API(aclrtReleaseMemAddress, addr);
  }
  allReserveMems_.clear();
  vmmMap_.clear();
}

namespace {
void MoveBackMappedHandle(std::map<DeviceMemPtr, aclrtDrvMemHandle> *mappedVmmHandle,
                          std::map<DeviceMemPtr, aclrtDrvMemHandle> *vmmMap,
                          std::set<aclrtDrvMemHandle> *cachedHandleSets_) {
  for (const auto [address, handle] : *mappedVmmHandle) {
    auto ret = CALL_ASCEND_API(aclrtUnmapMem, address);
    if (ret != ACL_SUCCESS) {
      LOG_ERROR << "Unmap memory failed, address : " << address << ".";
    } else {
      auto iter = vmmMap->find(address);
      if (iter == vmmMap->end()) {
        LOG_ERROR << "Find vmm map address : " << address << " failed.";
      } else {
        iter->second = nullptr;
        cachedHandleSets_->insert(handle);
      }
    }
  }
}
};  // namespace

size_t AscendVmmAdapter::MmapDeviceMem(const size_t size, const DeviceMemPtr addr, const size_t maxSize) {
  CHECK_IF_NULL(addr);
  LOG_OUT << "VMM MmapDeviceMem size:" << size << ", addr:" << addr
          << ", cachedHandleSets_ size : " << cachedHandleSets_.size() << ".";
  // use 0 temporarily
  auto local_rank_id = mrt::collective::CollectiveManager::Instance().local_rank_id();
  auto deviceId = local_rank_id;

  auto vmmStartAddr = FindVmmSegment(addr);
  if (vmmStartAddr == nullptr) {
    LOG_ERROR << "Can not find the vmm segment.";
    return 0;
  }
  aclrtPhysicalMemProp prop = {};
  prop.handleType = ACL_MEM_HANDLE_TYPE_NONE;
  prop.allocationType = ACL_MEM_ALLOCATION_TYPE_PINNED;
  prop.memAttr = ACL_HBM_MEM_HUGE;
  prop.location.type = ACL_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = deviceId;
  prop.reserve = 0;
  auto startOffset = CalAddressOffset(addr, vmmStartAddr);
  auto alignSize = GetRoundUpAlignSize(size + startOffset);
  auto handleSize = GetHandleSize(alignSize);
  auto iter = vmmMap_.find(vmmStartAddr);

  std::map<DeviceMemPtr, aclrtDrvMemHandle> mappedVmmHandle;
  for (size_t i = 0; i < handleSize; ++i) {
    auto newAddr = AddressOffset(vmmStartAddr, i * vmmAlignSize_);
    if (iter == vmmMap_.end() || iter->first != newAddr) {
      LOG_ERROR << "Can not find the vmm segment.";
      return 0;
    }
    if (iter->second != nullptr) {
      iter++;
      continue;
    }
    aclrtDrvMemHandle handle = nullptr;
    if (!cachedHandleSets_.empty()) {
      handle = *cachedHandleSets_.begin();
      cachedHandleSets_.erase(cachedHandleSets_.begin());
    } else {
      if (physicalHandleSize_ * vmmAlignSize_ >= maxSize) {
        LOG_OUT << "Mapped too much memory, physicalHandleSize_ : " << physicalHandleSize_ << ", maxSize : " << maxSize
                << ", addr : " << addr << ", size : " << size << ".";
        MoveBackMappedHandle(&mappedVmmHandle, &vmmMap_, &cachedHandleSets_);
        return 0;
      }

      auto ret = CALL_ASCEND_API(aclrtMallocPhysical, &handle, vmmAlignSize_, &prop, 0);
      if (ret != ACL_SUCCESS) {
        size_t usedHandleSize = 0;
        for (const auto &[k, v] : vmmMap_) {
          if (v != nullptr) {
            LOG_OUT << "Inuse handle address : " << k << ", handle : " << v << ".";
            usedHandleSize += 1;
          }
        }
        usedHandleSize += cachedHandleSets_.size();
        // This may be a normal case at the memory usage boundary.
        LOG_OUT << "Malloc physical memory failed, inuse physical memory handle size : " << usedHandleSize
                << ", physicalHandleSize_ size : " << physicalHandleSize_ << ".";
        MoveBackMappedHandle(&mappedVmmHandle, &vmmMap_, &cachedHandleSets_);
        return 0;
      } else {
        physicalHandleSize_++;
        if (physicalHandleSize_ * vmmAlignSize_ >= maxSize) {
          LOG_OUT << "Mapped too much memory, physicalHandleSize_ : " << physicalHandleSize_
                  << ", maxSize : " << maxSize << ".";
        }
      }
    }

    auto ret = CALL_ASCEND_API(aclrtMapMem, newAddr, vmmAlignSize_, 0, handle, 0);
    if (ret != ACL_SUCCESS) {
      LOG_ERROR << "Map memory failed.";
      cachedHandleSets_.insert(handle);
      MoveBackMappedHandle(&mappedVmmHandle, &vmmMap_, &cachedHandleSets_);
      return 0;
    }
    mappedVmmHandle[iter->first] = handle;
    iter->second = handle;
    iter++;
  }

  return size;
}

size_t AscendVmmAdapter::AllocDeviceMem(size_t size, DeviceMemPtr *addr) {
  CHECK_IF_NULL(addr);
  size_t alignSize = GetRoundUpAlignSize(size);
  LOG_OUT << "VMM AllocDeviceMem size:" << size << ", alignSize:" << alignSize;
  auto ret = CALL_ASCEND_API(aclrtReserveMemAddress, addr, alignSize, 0, nullptr, 1);
  if (ret != ACL_SUCCESS) {
    LOG_ERROR << "Reserve memory address failed.";
    return 0;
  }
  allReserveMems_.push_back(*addr);
  auto handleSize = GetHandleSize(alignSize);
  for (size_t i = 0; i < handleSize; i++) {
    auto newAddr = AddressOffset(*addr, i * vmmAlignSize_);
    vmmMap_[newAddr] = nullptr;
  }
  return alignSize;
}

size_t AscendVmmAdapter::EagerFreeDeviceMem(const DeviceMemPtr addr, const size_t size) {
  CHECK_IF_NULL(addr);
  LOG_OUT << "Eager free device mem addr :" << addr << ", size :" << size
          << ", cachedHandleSets_ size : " << cachedHandleSets_.size() << ".";
  size_t retSize = 0;
  auto iter = vmmMap_.lower_bound(addr);
  if (iter == vmmMap_.end()) {
    // Memory less than 2MB may be at the end of a vmm segment, and it's a normal case.
    if (size >= vmmAlignSize_) {
      LOG_ERROR << "Can not find the vmm segment.";
    }
    return 0;
  }
  auto vmmStartAddr = iter->first;
  auto freeEndAddr = AddressOffset(addr, size);
  while (true) {
    auto vmmEndAddr = AddressOffset(vmmStartAddr, vmmAlignSize_);
    if (vmmEndAddr > freeEndAddr) {
      break;
    }
    if (iter == vmmMap_.end() || iter->first != vmmStartAddr) {
      LOG_ERROR << "Can not find the vmm segment.";
      return 0;
    }
    if (iter->second == nullptr) {
      iter++;
      vmmStartAddr = vmmEndAddr;
      // Here nullptr may be huge, skip do logging.
      continue;
    }
    auto ret = CALL_ASCEND_API(aclrtUnmapMem, vmmStartAddr);
    if (ret != ACL_SUCCESS) {
      LOG_ERROR << "Unmap memory failed.";
      return 0;
    }
    cachedHandleSets_.insert(iter->second);
    iter->second = nullptr;
    iter++;
    vmmStartAddr = vmmEndAddr;
    retSize += vmmAlignSize_;
  }
  LOG_OUT << "After eager free, cachedHandleSets_ size : " << cachedHandleSets_.size()
          << ", expected free size : " << size << ", real size : " << retSize << ".";
  return retSize;
}

size_t AscendVmmAdapter::EmptyCache() {
  size_t emptySize = 0L;
  for (auto iter = cachedHandleSets_.begin(); iter != cachedHandleSets_.end(); iter++) {
    auto ret = CALL_ASCEND_API(aclrtFreePhysical, *iter);
    if (ret != ACL_SUCCESS) {
      LOG_ERROR << "Free physical memory failed.";
    }
  }

  size_t cacheHandleSize = cachedHandleSets_.size();
  physicalHandleSize_ -= cacheHandleSize;
  emptySize += cacheHandleSize * vmmAlignSize_;
  cachedHandleSets_.clear();
  LOG_OUT << "Empty cache size: " << emptySize << ", cached handle set size: " << cachedHandleSets_.size() << ".";
  return emptySize;
}
}  // namespace ascend
}  // namespace device
}  // namespace mrt
