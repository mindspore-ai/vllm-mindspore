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

#include "hardware/ascend/res_manager/mem_manager/ascend_memory_adapter.h"
#include "hardware/ascend/res_manager/mem_manager/ascend_dynamic_mem_adapter.h"
#include "hardware/ascend/res_manager/mem_manager/ascend_gmem_adapter.h"
#include "hardware/ascend/res_manager/mem_manager/ascend_vmm_adapter.h"
#include "hardware/ascend/res_manager/symbol_interface/acl_rt_symbol.h"
#include "hardware/ascend/res_manager/symbol_interface/symbol_utils.h"
#include "common/common.h"

namespace mrt {
namespace device {
namespace ascend {
namespace {
constexpr size_t kMBToByte = 1024 << 10;
constexpr size_t kGBToByte = 1024 << 20;
constexpr uint64_t kAscendMemAlignSize = 512;
constexpr double kHalfRatio = 0.5;
constexpr double kMSMemoryRatio = 0.9375;           // 15/16
constexpr double kReservedMemoryRatio = 0.0625;     // 1/16
constexpr size_t kPerHugePageMemorySize = 2097152;  // 2mb
constexpr size_t kExtraReservedMemory = 10485760;   // 10mb
constexpr size_t kSimuHBMTotalMemSizeGB = 64;
}  // namespace
AscendMemAdapterPtr AscendMemAdapter::instance_ = nullptr;

AscendMemAdapterPtr AscendMemAdapter::GetInstance() {
  if (instance_ == nullptr) {
    instance_ = std::make_shared<AscendDynamicMemAdapter>();
  }
  return instance_;
}

size_t AscendMemAdapter::GetRoundDownAlignSize(size_t inputSize) {
  return (inputSize / kAscendMemAlignSize) * kAscendMemAlignSize;
}

size_t AscendMemAdapter::GetRoundUpAlignSize(size_t inputSize) {
  return ((inputSize + kAscendMemAlignSize - 1) / kAscendMemAlignSize) * kAscendMemAlignSize;
}

size_t AscendMemAdapter::GetDeviceMemSizeFromContext() const {
  size_t sizeFromContext;
  float totalDeviceMemory = 32.0f;
  auto maxDeviceMemory = totalDeviceMemory;
  // if (context->ascend_soc_version() == kAscendVersion910b || context->ascend_soc_version() == kAscendVersion910_93) {
  //   totalDeviceMemory = 64.0f;
  // }
  // if (context->ascend_soc_version() == kAscendVersion310p) {
  //   totalDeviceMemory = 43.0f;
  // }
  LOG_OUT << "context maxDeviceMemory:" << maxDeviceMemory;
  sizeFromContext = FloatToSize(maxDeviceMemory * kGBToByte);

  return sizeFromContext;
}

bool AscendMemAdapter::Initialize() {
  if (initialized_) {
    return true;
  }

  // use 0 temporarily.
  float hugePageReserveSize = 0;
  deviceHbmHugePageReservedSize_ = static_cast<size_t>(hugePageReserveSize * kGBToByte);
  if (AscendVmmAdapter::IsEnabled() && deviceHbmHugePageReservedSize_ > 0) {
    LOG_OUT << "Reserve huge page feature is not available when VMM is enabled.";
  }
  LOG_OUT << "Config hugePageReserveSize : " << hugePageReserveSize
          << ", deviceHbmHugePageReservedSize_ : " << deviceHbmHugePageReservedSize_;

  auto ret = CALL_ASCEND_API(aclrtGetMemInfo, ACL_HBM_MEM, &deviceHbmFreeSize_, &deviceHbmTotalSize_);
  if (ret != ACL_SUCCESS || deviceHbmTotalSize_ == 0) {
    LOG_ERROR << "Internal Error: Get Device MOC memory size failed, ret = " << ret
              << ", total MOC size :" << deviceHbmTotalSize_;
  }

  if (deviceHbmFreeSize_ < LongToSize(DoubleToLong(deviceHbmTotalSize_ * kHalfRatio))) {
    // use 0 temporarily.
    unsigned int deviceId = 0;
    LOG_OUT << "Free memory size is less "
               "than half of total memory size."
            << "Device " << deviceId << " Device MOC total size:" << deviceHbmTotalSize_
            << " Device MOC free size:" << deviceHbmFreeSize_
            << " may be other processes occupying this card, check as: ps -ef|grep python";
  }

  // get user define max backend memory
  auto userDefineMsSize = GetDeviceMemSizeFromContext();
  auto recommendMemSizeForOthers = LongToSize(DoubleToLong(deviceHbmFreeSize_ * kReservedMemoryRatio));
  size_t reservedMemSizeForOthers;
  if (userDefineMsSize == 0) {
    msUsedHbmSize_ = DoubleToLong(deviceHbmFreeSize_ * kMSMemoryRatio);
    // sub the extra reserved 10mb after rounding down the 2mb
    msUsedHbmSize_ = (msUsedHbmSize_ / kPerHugePageMemorySize) * kPerHugePageMemorySize - kExtraReservedMemory;
    reservedMemSizeForOthers = deviceHbmFreeSize_ - SizeToLong(msUsedHbmSize_);
  } else {
    if (userDefineMsSize >= deviceHbmFreeSize_) {
      LOG_ERROR << "#umsg#Framework Error Message:#umsg#The Free Device Memory Size is "
                << (SizeToFloat(deviceHbmFreeSize_) / kGBToByte) << " GB, maxDeviceMemory should be in range (0-"
                << (SizeToFloat(deviceHbmFreeSize_) / kMBToByte) << "]MB, but got "
                << (SizeToFloat(userDefineMsSize) / kMBToByte)
                << "MB, please set the context key maxDeviceMemory in valid range.";
    }
    msUsedHbmSize_ = SizeToLong(userDefineMsSize);

    reservedMemSizeForOthers = deviceHbmTotalSize_ - LongToSize(msUsedHbmSize_);
    if (reservedMemSizeForOthers < recommendMemSizeForOthers) {
      LOG_OUT << "Reserved memory size for other components(" << reservedMemSizeForOthers
              << ") is less than recommend size(" << recommendMemSizeForOthers
              << "), It may lead to Out Of Memory in HCCL or other components, Please double check context key "
                 "'variable_memory_max_size'/'maxDeviceMemory'";
    }
  }

  if (AscendVmmAdapter::GetInstance().IsEnabled()) {
    msUsedHbmSize_ = SizeToLong(AscendVmmAdapter::GetInstance().GetRoundDownAlignSize(msUsedHbmSize_));
  } else if (AscendGmemAdapter::GetInstance().is_eager_free_enabled()) {
    msUsedHbmSize_ = SizeToLong(AscendGmemAdapter::GetInstance().GetRoundDownAlignSize(msUsedHbmSize_));
  } else {
    msUsedHbmSize_ = SizeToLong(GetRoundDownAlignSize(msUsedHbmSize_));
  }
  maxAvailableMsHbmSize_ = msUsedHbmSize_;

  auto getInitInfo = [this, &reservedMemSizeForOthers, &recommendMemSizeForOthers, &userDefineMsSize]() -> std::string {
    std::ostringstream oss;
    oss << "Device MOC Size:" << deviceHbmTotalSize_ / kMBToByte
        << "M, Device free MOC Size:" << deviceHbmFreeSize_ / kMBToByte
        << "M, Reserved MOC size for Other Components(HCCL/rts/etc.):" << reservedMemSizeForOthers / kMBToByte
        << "M, Recommend Reserved MOC size for Other Components:" << recommendMemSizeForOthers / kMBToByte
        << "M, User define inferrt MOC Size:" << userDefineMsSize / kGBToByte
        << "G, inferrt Used MOC Size:" << msUsedHbmSize_ / kMBToByte << "M.";
    return oss.str();
  };

  LOG_OUT << getInitInfo();
  initialized_ = true;
  return true;
}

void AscendMemAdapter::SimulationInitialize() {
  deviceHbmTotalSize_ = kSimuHBMTotalMemSizeGB * kGBToByte;
  deviceHbmFreeSize_ = deviceHbmTotalSize_;
  size_t reservedMemSizeForOthers;
  auto userDefineMsSize = GetDeviceMemSizeFromContext();
  if (userDefineMsSize == 0) {
    msUsedHbmSize_ = DoubleToLong(deviceHbmFreeSize_ * kMSMemoryRatio);
    msUsedHbmSize_ = (msUsedHbmSize_ / kPerHugePageMemorySize) * kPerHugePageMemorySize - kExtraReservedMemory;
    reservedMemSizeForOthers = deviceHbmFreeSize_ - SizeToLong(msUsedHbmSize_);
  } else {
    msUsedHbmSize_ = SizeToLong(userDefineMsSize);
    if (userDefineMsSize > deviceHbmTotalSize_) {
      deviceHbmTotalSize_ = userDefineMsSize;
    }
    reservedMemSizeForOthers = deviceHbmTotalSize_ - userDefineMsSize;
  }

  LOG_OUT << "Simulation Device MOC Size:" << deviceHbmTotalSize_ / kMBToByte
          << "M, Device free MOC Size:" << deviceHbmFreeSize_ / kMBToByte
          << "M, Reserved MOC size for Other Components(HCCL/rts/etc.):" << reservedMemSizeForOthers / kMBToByte
          << "M, User define inferrt MOC Size:" << userDefineMsSize / kGBToByte
          << "G, inferrt Used MOC Size:" << msUsedHbmSize_ / kMBToByte << "M.";
  maxAvailableMsHbmSize_ = msUsedHbmSize_;
  initialized_ = true;
}

bool AscendMemAdapter::DeInitialize() {
  if (!initialized_) {
    LOG_OUT << "DeInitialize Ascend Memory Adapter when it is not initialize";
    return false;
  }
  std::ostringstream ossBuf;
  ossBuf << "Ascend Memory Adapter deinitialize success, statistics:" << DevMemStatistics();
  LOG_OUT << ossBuf.str();
  deviceHbmTotalSize_ = 0;
  deviceHbmFreeSize_ = 0;
  msUsedHbmSize_ = 0;
  maxAvailableMsHbmSize_ = 0;
  initialized_ = false;
  return true;
}

namespace {
struct HugeMemReserver {
  HugeMemReserver(size_t size, size_t reserverSize) {
    LOG_OUT << "Allocate size : " << size << ", reserve_size : " << reserverSize << ".";
    if (reserverSize < kMBToByte) {
      return;
    }
    size_t freeSize = 0;
    size_t totalSize = 0;
    auto ret = CALL_ASCEND_API(aclrtGetMemInfo, ACL_HBM_MEM_HUGE, &freeSize, &totalSize);
    LOG_OUT << "Huge mem reserve freeSize : " << freeSize << ", totalSize : " << totalSize << ".";
    if (ret == ACL_SUCCESS) {
      if (freeSize < reserverSize + size) {
        LOG_OUT << "Free size of huge page mem[" << freeSize
                << "] is less than the sum of reserverSize and allocate size. Reserve size " << reserverSize
                << ", allocate size : " << size << ", total ACL_HBM_MEM_HUGE size : " << totalSize << ".";
        if (freeSize < reserverSize) {
          LOG_ERROR << "Free size of huge page mem[" << freeSize << "] is less than reserverSize : " << reserverSize
                    << ", change reserve operation with free size.";
          reserverSize = freeSize;
        }
        ret = CALL_ASCEND_API(aclrtMalloc, reinterpret_cast<void **>(&addr_), reserverSize, ACL_MEM_MALLOC_HUGE_ONLY);
        if (ret != ACL_RT_SUCCESS) {
          addr_ = nullptr;
          LOG_ERROR << "aclrtMalloc mem size[" << reserverSize << "] fail, ret[" << ret << "]";
        } else {
          LOG_OUT << "Huge mem reserve success, addr : " << addr_ << ", size : " << reserverSize << ".";
        }
      }
    } else {
      LOG_OUT << "aclrtGetMemInfo mem size[" << size << "] fail, ret[" << ret << "]";
    }
  }

  ~HugeMemReserver() {
    if (addr_ != nullptr) {
      auto ret = CALL_ASCEND_API(aclrtFree, addr_);
      if (ret != ACL_SUCCESS) {
        LOG_ERROR << "aclrtFree mem [" << addr_ << "] fail, ret[" << ret << "]";
      } else {
        LOG_OUT << "Huge mem reserve success, free : " << addr_ << ".";
      }
    }
  }

  void *addr_{nullptr};
};
}  // namespace

uint8_t *AscendMemAdapter::MallocFromRts(size_t size) const {
  uint8_t *ptr = nullptr;
  if (AscendVmmAdapter::GetInstance().IsEnabled()) {
    return nullptr;
  }
  if (AscendGmemAdapter::GetInstance().is_eager_free_enabled()) {
    return AscendGmemAdapter::GetInstance().MmapMemory(size, reinterpret_cast<void *>(ptr));
  }

  HugeMemReserver huge_mem_reserver(size, deviceHbmHugePageReservedSize_);
  auto ret = CALL_ASCEND_API(aclrtMalloc, reinterpret_cast<void **>(&ptr), size, ACL_MEM_TYPE_HIGH_BAND_WIDTH);
  if (ret != ACL_RT_SUCCESS) {
    if (ret == ACL_ERROR_RT_MEMORY_ALLOCATION) {
      // use 0 temporarily.
      unsigned int deviceId = 0;
      size_t freeSize = 0;
      size_t total = 0;
      (void)CALL_ASCEND_API(aclrtGetMemInfo, ACL_HBM_MEM, &freeSize, &total);
      LOG_ERROR << "#umsg#Framework Error Message:#umsg#Malloc device memory failed, size[" << size << "], ret[" << ret
                << "], "
                << "Device " << deviceId << " Available MOC size:" << total << " free size:" << freeSize
                << " may be other processes occupying this card, check as: ps -ef|grep python";
    } else {
      LOG_ERROR << "rtMalloc mem size[" << size << "] fail, ret[" << ret << "]";
    }
  } else {
    LOG_OUT << "Call rtMalloc to allocate device memory Success, size: " << size
            << " bytes, address start: " << reinterpret_cast<void *>(ptr)
            << " end: " << reinterpret_cast<void *>(ptr + size);
  }
  return ptr;
}

bool AscendMemAdapter::FreeToRts(void *devPtr, const size_t size) const {
  if (devPtr != nullptr) {
    if (AscendGmemAdapter::GetInstance().is_eager_free_enabled()) {
      return AscendGmemAdapter::GetInstance().MunmapMemory(devPtr, size);
    }
    auto ret = CALL_ASCEND_API(aclrtFree, devPtr);
    if (ret != ACL_SUCCESS) {
      LOG_ERROR << "aclrtFree mem [" << devPtr << "] fail, ret[" << ret << "]";
      return false;
    }
  }
  return true;
}
}  // namespace ascend
}  // namespace device
}  // namespace mrt
