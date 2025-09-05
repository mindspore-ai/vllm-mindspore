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
#include "hardware/ascend/res_manager/mem_manager/ascend_gmem_adapter.h"
#include <pthread.h>
#include <sys/types.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <tuple>
#include "common/common.h"
#include "hardware/ascend/res_manager/ascend_stream_manager.h"

namespace mrt {
namespace device {
namespace ascend {
static constexpr const char kGMemLibName[] = "libgmem.so";
static constexpr const char kMsEnableGmem[] = "MS_ENABLE_GMEM";
constexpr uint64_t kAscendMmapAlignSize = 1 << 21;
constexpr int kMapPeerShared = 0x8000000;

const size_t AscendGmemAdapter::GetRoundUpAlignSize(size_t inputSize) const {
  return (inputSize + kAscendMmapAlignSize - 1) & ~(kAscendMmapAlignSize - 1);
}

const size_t AscendGmemAdapter::GetRoundDownAlignSize(size_t inputSize) const {
  return inputSize & ~(kAscendMmapAlignSize - 1);
}

size_t AscendGmemAdapter::AllocDeviceMem(size_t size, DeviceMemPtr *addr) const {
  size_t alignSize = GetRoundUpAlignSize(size);
  uint8_t *allocAddr = MmapMemory(alignSize, nullptr);
  if (allocAddr == nullptr) {
    LOG_OUT << "Malloc memory failed.";
    return 0;
  }
  *addr = allocAddr;
  return alignSize;
}

size_t AscendGmemAdapter::EagerFreeDeviceMem(const DeviceMemPtr addr, const size_t size) const {
  CHECK_IF_NULL(addr);
  LOG_OUT << "Enter ascend eager free device mem, addr : " << addr << ", size : " << size << ".";
  if (size == 0) {
    LOG_OUT << "Eager free device mem, addr : " << addr << ", size is zero.";
    return 0;
  }
  size_t addrSizeT = reinterpret_cast<size_t>(addr);
  // Adjust addr -> round up addr, size -> round down size.
  size_t fromAddr = GetRoundUpAlignSize(addrSizeT);
  size_t endAddr = GetRoundDownAlignSize(addrSizeT + size);
  if (endAddr <= fromAddr) {
    LOG_OUT << "End addr : " << endAddr << " is not bigger than fromAddr : " << fromAddr << ".";
    return 0;
  }
  size_t realSize = endAddr - fromAddr;
  int ret = freeEager_(fromAddr, SizeToUlong(realSize), nullptr);
  return ret != 0 ? 0 : realSize;
}

uint8_t *AscendGmemAdapter::MmapMemory(size_t size, void *addr) const {
  LOG_OUT << "Enter mmap memory, size : " << size << ".";
  if (size == 0) {
    LOG_ERROR << "Mmap memory, addr : " << addr << ", size is zero.";
    return nullptr;
  }

  int flags = MAP_PRIVATE | MAP_ANONYMOUS | kMapPeerShared;
  int prot = PROT_READ | PROT_WRITE;
  void *mappedAddr = mmap(addr, size, prot, flags, -1, 0);
  if (mappedAddr == MAP_FAILED) {
    LOG_ERROR << "Mmap failed.";
  }
  return static_cast<uint8_t *>(mappedAddr);
}

bool AscendGmemAdapter::MunmapMemory(void *addr, const size_t size) const {
  LOG_OUT << "Enter munmap memory, addr : " << addr << ", size : " << size << ".";
  auto ret = munmap(addr, size);
  return ret != -1;
}

void AscendGmemAdapter::LoadGMemLib() noexcept {
  LOG_OUT << "MS_ENABLE_GMEM is set, try to open gmem.";
  gmemHandle_ = dlopen(kGMemLibName, RTLD_NOW);
  if (gmemHandle_ != nullptr) {
    LOG_OUT << "Open GMem lib success, inferrt will use gmem to optimize memory usage.";
    LIB_FUNC(GMEM_FREE_EAGER) gmemFreeEager = DlsymFuncObj(gmemFreeEager, gmemHandle_);
    if (gmemFreeEager != nullptr) {
      isEagerFreeEnabled_ = true;
      freeEager_ = gmemFreeEager;
    } else {
      LOG_OUT << "Load gmem free eager failed.";
      if (dlclose(gmemHandle_) != 0) {
        LOG_ERROR << "Close GMem lib failed, detail : " << dlerror() << ".";
      }
    }
  } else {
    LOG_OUT << "Open GMem lib failed.";
  }
}

void AscendGmemAdapter::UnloadGMemLib() noexcept {
  if (gmemHandle_ != nullptr) {
    LOG_OUT << "Close GMem lib.";
    if (dlclose(gmemHandle_) != 0) {
      LOG_ERROR << "Close GMem lib failed, detail : " << dlerror() << ".";
    }
    gmemHandle_ = nullptr;
  }
}
}  // namespace ascend
}  // namespace device
}  // namespace mrt
