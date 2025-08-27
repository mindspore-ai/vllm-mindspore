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

namespace mrt {
namespace device {
namespace ascend {
size_t AscendVmmAdapter::GetRoundUpAlignSize(size_t input_size) const {
  return ((input_size + vmmAlignSize_ - 1) / vmmAlignSize_) * vmmAlignSize_;
}

size_t AscendVmmAdapter::GetRoundDownAlignSize(size_t input_size) const {
  return (input_size / vmmAlignSize_) * vmmAlignSize_;
}

size_t AscendVmmAdapter::GetHandleSize(size_t input_size) {
  if (input_size % vmmAlignSize_ != 0) {
    LOG_ERROR << "Input size must be multiple of 2MB, but got " << input_size;
  }
  return input_size / vmmAlignSize_;
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
void MoveBackMappedHandle(std::map<DeviceMemPtr, aclrtDrvMemHandle> *mapped_vmm_handle,
                          std::map<DeviceMemPtr, aclrtDrvMemHandle> *vmm_map,
                          std::set<aclrtDrvMemHandle> *cachedHandleSets_) {
  for (const auto [address, handle] : *mapped_vmm_handle) {
    auto ret = CALL_ASCEND_API(aclrtUnmapMem, address);
    if (ret != ACL_SUCCESS) {
      LOG_ERROR << "Unmap memory failed, address : " << address << ".";
    } else {
      auto iter = vmm_map->find(address);
      if (iter == vmm_map->end()) {
        LOG_ERROR << "Find vmm map address : " << address << " failed.";
      } else {
        iter->second = nullptr;
        cachedHandleSets_->insert(handle);
      }
    }
  }
}
};  // namespace

size_t AscendVmmAdapter::MmapDeviceMem(const size_t size, const DeviceMemPtr addr, const size_t max_size) {
  CHECK_IF_NULL(addr);
  LOG_OUT << "VMM MmapDeviceMem size:" << size << ", addr:" << addr
          << ", cachedHandleSets_ size : " << cachedHandleSets_.size() << ".";
  // use 0 temporarily
  auto device_id = 0;

  auto vmm_start_addr = FindVmmSegment(addr);
  if (vmm_start_addr == nullptr) {
    LOG_ERROR << "Can not find the vmm segment.";
    return 0;
  }
  aclrtPhysicalMemProp prop = {};
  prop.handleType = ACL_MEM_HANDLE_TYPE_NONE;
  prop.allocationType = ACL_MEM_ALLOCATION_TYPE_PINNED;
  prop.memAttr = ACL_HBM_MEM_HUGE;
  prop.location.type = ACL_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = device_id;
  prop.reserve = 0;
  auto start_offset = CalAddressOffset(addr, vmm_start_addr);
  auto align_size = GetRoundUpAlignSize(size + start_offset);
  auto handle_size = GetHandleSize(align_size);
  auto iter = vmmMap_.find(vmm_start_addr);

  std::map<DeviceMemPtr, aclrtDrvMemHandle> mapped_vmm_handle;
  for (size_t i = 0; i < handle_size; ++i) {
    auto new_addr = AddressOffset(vmm_start_addr, i * vmmAlignSize_);
    if (iter == vmmMap_.end() || iter->first != new_addr) {
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
      if (physicalHandleSize_ * vmmAlignSize_ >= max_size) {
        LOG_OUT << "Mapped too much memory, physicalHandleSize_ : " << physicalHandleSize_
                << ", max_size : " << max_size << ", addr : " << addr << ", size : " << size << ".";
        MoveBackMappedHandle(&mapped_vmm_handle, &vmmMap_, &cachedHandleSets_);
        return 0;
      }

      auto ret = CALL_ASCEND_API(aclrtMallocPhysical, &handle, vmmAlignSize_, &prop, 0);
      if (ret != ACL_SUCCESS) {
        size_t used_handle_size = 0;
        for (const auto &[k, v] : vmmMap_) {
          if (v != nullptr) {
            LOG_OUT << "Inuse handle address : " << k << ", handle : " << v << ".";
            used_handle_size += 1;
          }
        }
        used_handle_size += cachedHandleSets_.size();
        // This may be a normal case at the memory usage boundary.
        LOG_OUT << "Malloc physical memory failed, inuse physical memory handle size : " << used_handle_size
                << ", physicalHandleSize_ size : " << physicalHandleSize_ << ".";
        MoveBackMappedHandle(&mapped_vmm_handle, &vmmMap_, &cachedHandleSets_);
        return 0;
      } else {
        physicalHandleSize_++;
        if (physicalHandleSize_ * vmmAlignSize_ >= max_size) {
          LOG_OUT << "Mapped too much memory, physicalHandleSize_ : " << physicalHandleSize_
                  << ", max_size : " << max_size << ".";
        }
      }
    }

    auto ret = CALL_ASCEND_API(aclrtMapMem, new_addr, vmmAlignSize_, 0, handle, 0);
    if (ret != ACL_SUCCESS) {
      LOG_ERROR << "Map memory failed.";
      cachedHandleSets_.insert(handle);
      MoveBackMappedHandle(&mapped_vmm_handle, &vmmMap_, &cachedHandleSets_);
      return 0;
    }
    mapped_vmm_handle[iter->first] = handle;
    iter->second = handle;
    iter++;
  }

  return size;
}

size_t AscendVmmAdapter::AllocDeviceMem(size_t size, DeviceMemPtr *addr) {
  CHECK_IF_NULL(addr);
  size_t align_size = GetRoundUpAlignSize(size);
  LOG_OUT << "VMM AllocDeviceMem size:" << size << ", align_size:" << align_size;
  auto ret = CALL_ASCEND_API(aclrtReserveMemAddress, addr, align_size, 0, nullptr, 1);
  if (ret != ACL_SUCCESS) {
    LOG_ERROR << "Reserve memory address failed.";
    return 0;
  }
  allReserveMems_.push_back(*addr);
  auto handle_size = GetHandleSize(align_size);
  for (size_t i = 0; i < handle_size; i++) {
    auto new_addr = AddressOffset(*addr, i * vmmAlignSize_);
    vmmMap_[new_addr] = nullptr;
  }
  return align_size;
}

size_t AscendVmmAdapter::EagerFreeDeviceMem(const DeviceMemPtr addr, const size_t size) {
  CHECK_IF_NULL(addr);
  LOG_OUT << "Eager free device mem addr :" << addr << ", size :" << size
          << ", cachedHandleSets_ size : " << cachedHandleSets_.size() << ".";
  size_t ret_size = 0;
  auto iter = vmmMap_.lower_bound(addr);
  if (iter == vmmMap_.end()) {
    // Memory less than 2MB may be at the end of a vmm segment, and it's a normal case.
    if (size >= vmmAlignSize_) {
      LOG_ERROR << "Can not find the vmm segment.";
    }
    return 0;
  }
  auto vmm_start_addr = iter->first;
  auto free_end_addr = AddressOffset(addr, size);
  while (true) {
    auto vmm_end_addr = AddressOffset(vmm_start_addr, vmmAlignSize_);
    if (vmm_end_addr > free_end_addr) {
      break;
    }
    if (iter == vmmMap_.end() || iter->first != vmm_start_addr) {
      LOG_ERROR << "Can not find the vmm segment.";
      return 0;
    }
    if (iter->second == nullptr) {
      iter++;
      vmm_start_addr = vmm_end_addr;
      // Here nullptr may be huge, skip do logging.
      continue;
    }
    auto ret = CALL_ASCEND_API(aclrtUnmapMem, vmm_start_addr);
    if (ret != ACL_SUCCESS) {
      LOG_ERROR << "Unmap memory failed.";
      return 0;
    }
    cachedHandleSets_.insert(iter->second);
    iter->second = nullptr;
    iter++;
    vmm_start_addr = vmm_end_addr;
    ret_size += vmmAlignSize_;
  }
  LOG_OUT << "After eager free, cachedHandleSets_ size : " << cachedHandleSets_.size()
          << ", expected free size : " << size << ", real size : " << ret_size << ".";
  return ret_size;
}

size_t AscendVmmAdapter::EmptyCache() {
  size_t empty_size = 0L;
  for (auto iter = cachedHandleSets_.begin(); iter != cachedHandleSets_.end(); iter++) {
    auto ret = CALL_ASCEND_API(aclrtFreePhysical, *iter);
    if (ret != ACL_SUCCESS) {
      LOG_ERROR << "Free physical memory failed.";
    }
  }

  size_t cache_handle_size = cachedHandleSets_.size();
  physicalHandleSize_ -= cache_handle_size;
  empty_size += cache_handle_size * vmmAlignSize_;
  cachedHandleSets_.clear();
  LOG_OUT << "Empty cache size: " << empty_size << ", cached handle set size: " << cachedHandleSets_.size() << ".";
  return empty_size;
}
}  // namespace ascend
}  // namespace device
}  // namespace mrt
