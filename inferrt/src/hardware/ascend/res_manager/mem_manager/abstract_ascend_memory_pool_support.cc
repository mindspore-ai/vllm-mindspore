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

#include "hardware/ascend/res_manager/mem_manager/abstract_ascend_memory_pool_support.h"

#include <algorithm>
#include <utility>

#include "hardware/ascend/res_manager/mem_manager/ascend_memory_adapter.h"
#include "hardware/ascend/res_manager/mem_manager/ascend_gmem_adapter.h"
#include "hardware/ascend/res_manager/mem_manager/ascend_vmm_adapter.h"
#include "hardware/ascend/res_manager/ascend_stream_manager.h"
#include "common/common.h"
#include "hardware/ascend/res_manager/symbol_interface/acl_rt_symbol.h"
#include "hardware/ascend/res_manager/symbol_interface/symbol_utils.h"

namespace mrt {
namespace device {
namespace ascend {
// The minimum unit size (8MB) of memory block used for dynamic extend in graph run mode.
static const size_t ASCEND_COMMON_POOL_ALLOC_UNIT_SIZE_FOR_GRAPH_RUN_MODE = 8 << 20;
constexpr char kGlobalOverflowWorkspace[] = "GLOBAL_OVERFLOW_WORKSPACE";

void AbstractAscendMemoryPoolSupport::SetMemPoolBlockSize(size_t available_device_mem_size) {
  // set by default configuration
  SetMemAllocUintSize(kDynamicMemAllocUnitSize, kDynamicMemAllocUnitSize);
}

namespace {
bool NoAdditionalMemory() {
  // use default temporarily.
  return true;
}
}  // namespace

size_t AbstractAscendMemoryPoolSupport::CalMemBlockAllocSize(size_t size, bool from_persistent_mem, bool need_recycle) {
  auto device_free_mem_size = free_mem_size();
  if (device_free_mem_size < size) {
    LOG_OUT << "The device memory is not enough, the free memory size is " << device_free_mem_size
            << ", but the alloc size is " << size;
    LOG_OUT << "The dynamic memory pool total size is " << TotalMemStatistics() / kMBToByte << "M, total used size is "
            << TotalUsedMemStatistics() / kMBToByte << "M, used peak size is " << UsedMemPeakStatistics() / kMBToByte
            << "M.";
    LOG_OUT << "Memory Statistics:" << AscendMemAdapter::GetInstance()->DevMemStatistics();
    return 0;
  }

  size_t alloc_mem_size;
  SetMemPoolBlockSize(device_free_mem_size);
  auto alloc_mem_unit_size = MemAllocUnitSize(from_persistent_mem);
  if (need_recycle) {
    alloc_mem_unit_size = kDynamicMemAllocUnitSize;
  }
  LOG_OUT << "Get unit block size " << alloc_mem_unit_size;
  alloc_mem_size = alloc_mem_unit_size;

  const bool is_graph_run_mode = true;
  if (is_graph_run_mode) {
    // Growing at adding alloc unit size
    while (alloc_mem_size < size) {
      alloc_mem_size = alloc_mem_size + alloc_mem_unit_size;
    }
  } else {
    // Growing at twice of alloc unit size
    constexpr size_t kDouble = 2;
    while (alloc_mem_size < size) {
      alloc_mem_size = alloc_mem_size * kDouble;
    }
  }

  alloc_mem_size = std::min(alloc_mem_size, device_free_mem_size);
  if (NoAdditionalMemory() && !need_recycle) {
    alloc_mem_size = std::min(alloc_mem_size, size);
  }
  return alloc_mem_size;
}

size_t AbstractAscendMemoryPoolSupport::AllocDeviceMem(size_t size, DeviceMemPtr *addr) {
  LOG_OUT << "Malloc Memory for Pool, size: " << size;
  if (size == 0) {
    LOG_ERROR << "Failed to alloc memory pool resource, the size is zero!";
  }
  *addr = AscendMemAdapter::GetInstance()->MallocStaticDevMem(size);
  if (*addr == nullptr) {
    LOG_ERROR << "Alloc device memory pool address is nullptr, failed to alloc memory pool resource!";
  }
  return size;
}

size_t AbstractAscendMemoryPoolSupport::GetMaxUsedMemSize() const {
  void *min_used_addr = GetMinUsingMemoryAddr();
  if (min_used_addr == nullptr) {
    return 0;
  }
  return AscendMemAdapter::GetInstance()->GetDynamicMemUpperBound(min_used_addr);
}

size_t AbstractAscendMemoryPoolSupport::GetVmmUsedMemSize() const {
  if (IsEnableVmm()) {
    return AscendVmmAdapter::GetInstance().GetAllocatedSize();
  }
  return 0;
}

const bool AbstractAscendMemoryPoolSupport::IsEnableEagerFree() const {
  return AscendGmemAdapter::GetInstance().is_eager_free_enabled();
}

const bool AbstractAscendMemoryPoolSupport::SyncAllStreams() { return AscendStreamMng::GetInstance().SyncAllStreams(); }

size_t AbstractAscendMemoryPoolSupport::AllocDeviceMemByEagerFree(size_t size, DeviceMemPtr *addr) {
  if (IsEnableVmm()) {
    return AscendVmmAdapter::GetInstance().AllocDeviceMem(size, addr);
  } else if (IsEnableEagerFree()) {
    return AscendGmemAdapter::GetInstance().AllocDeviceMem(size, addr);
  } else {
    LOG_ERROR << "Eager free and VMM are both disabled.";
    return 0;
  }
}

size_t AbstractAscendMemoryPoolSupport::FreeDeviceMemByEagerFree(const DeviceMemPtr addr, const size_t size) {
  if (IsEnableVmm()) {
    return AscendVmmAdapter::GetInstance().EagerFreeDeviceMem(addr, size);
  } else if (IsEnableEagerFree()) {
    return AscendGmemAdapter::GetInstance().EagerFreeDeviceMem(addr, size);
  } else {
    LOG_ERROR << "Eager free and VMM are both disabled.";
    return 0;
  }
}

size_t AbstractAscendMemoryPoolSupport::EmptyCache() { return AscendVmmAdapter::GetInstance().EmptyCache(); }

size_t AbstractAscendMemoryPoolSupport::MmapDeviceMem(const size_t size, const DeviceMemPtr addr) {
  if (IsEnableVmm()) {
    return AscendVmmAdapter::GetInstance().MmapDeviceMem(size, addr, total_mem_size());
  } else if (IsEnableEagerFree()) {
    auto ret = AscendGmemAdapter::GetInstance().MmapMemory(size, addr);
    if (ret == nullptr) {
      LOG_ERROR << "Mmap memory failed.";
    }
    return size;
  }
  LOG_ERROR << "Eager free and VMM are both disabled.";
  return 0;
}

bool AbstractAscendMemoryPoolSupport::FreeDeviceMem(const DeviceMemPtr &addr) {
  CHECK_IF_NULL(addr);
  int64_t max_actual = ActualPeakStatistics();
  LOG_OUT << "Max actual used memory size is " << max_actual;
  AscendMemAdapter::GetInstance()->UpdateActualPeakMemory(max_actual);
  int64_t max_peak = UsedMemPeakStatistics();
  LOG_OUT << "Max peak used memory size is " << max_peak;
  AscendMemAdapter::GetInstance()->UpdateUsedPeakMemory(max_peak);
  // disable ge kernel use two pointer mem adapter, not support free.
  // if (!IsEnableVmm() && !IsEnableEagerFree() && !IsDisableGeKernel()) {
  //   return AscendMemAdapter::GetInstance()->FreeStaticDevMem(addr);
  // }
  return true;
}

void AbstractAscendMemoryPoolSupport::ResetIdleMemBuf() const {
  // Warning : This method is not in used currently, removed in next release.
}

size_t AbstractAscendMemoryPoolSupport::free_mem_size() { return AscendMemAdapter::GetInstance()->FreeDevMemSize(); }

uint64_t AbstractAscendMemoryPoolSupport::total_mem_size() const {

  return AscendMemAdapter::GetInstance()->MaxHbmSizeForMs();
}
}  // namespace ascend
}  // namespace device
}  // namespace mrt
