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

#ifndef __RUNTIME_EXECUTOR_KERNEL_LAUNCH_GROUP_MEMORY_CACHE_H__
#define __RUNTIME_EXECUTOR_KERNEL_LAUNCH_GROUP_MEMORY_CACHE_H__

#include <memory>
#include <map>
#include <vector>
#include <unordered_map>
#include "runtime/utils/spinlock.h"
#include "hardware/hardware_abstract/device_context.h"
#include "ir/tensor/tensor.h"

namespace mrt {
namespace runtime {
using device::DeviceContext;
using ir::Tensor;
class OpRunner;

enum MemoryType {
  kOutputMem = 0,
  kWorkspaceMem = 1,
};

struct KernelMemoryTraceBlock {
  KernelMemoryTraceBlock(OpRunner *kernel, void *start, size_t size, MemoryType memType, size_t index, Tensor *tensor,
                         const DeviceContext *deviceContext)
      : op_(kernel),
        start_(reinterpret_cast<uint8_t *>(start)),
        end_(reinterpret_cast<uint8_t *>(start) + size),
        size_(size),
        memType_(memType),
        index_(index),
        tensor_(tensor),
        memoryTraceBlockIndex_(0),
        offsetInMemoryTraceBlock_(0),
        deviceContext_(deviceContext) {}

  OpRunner *op_;
  uint8_t *start_;
  uint8_t *end_;
  size_t size_;
  MemoryType memType_;
  size_t index_;
  Tensor *tensor_;

  size_t memoryTraceBlockIndex_;
  size_t offsetInMemoryTraceBlock_;
  const DeviceContext *deviceContext_;
  SpinLock lock_;
};

struct MemoryTraceBlock {
  MemoryTraceBlock(uint8_t *start, size_t size) : start_(start), end_(start + size), size_(size) {}

  uint8_t *start_;
  uint8_t *end_;
  size_t size_;
};

using KernelMemoryTraceBlockPtr = std::shared_ptr<KernelMemoryTraceBlock>;
using MemoryTraceBlockPtr = std::shared_ptr<MemoryTraceBlock>;

class MemoryCache {
 public:
  MemoryCache();
  ~MemoryCache() = default;

  void ReserveKernelMemoryBlocks(size_t size, const DeviceContext *deviceContext);

  void AddKernelMemoryTraceBlock(const KernelMemoryTraceBlockPtr &block, const DeviceContext *deviceContext);

  const std::shared_ptr<std::map<const DeviceContext *, std::vector<MemoryTraceBlockPtr>>> &GetMergeBlocks() const;

  const std::shared_ptr<std::unordered_map<OpRunner *, std::vector<KernelMemoryTraceBlockPtr>>> &
  GetAllKernelBlocksInfo();

  const std::shared_ptr<std::unordered_map<const Tensor *, KernelMemoryTraceBlockPtr>> &GetTensorToMemBlocksInfo()
    const;

  void MergeBlocks();

  void ClearExpiredCache();

  void ClearAllCache();

 private:
  DISABLE_COPY_AND_ASSIGN(MemoryCache);

  void MergeBlocksForSameDeviceContext(std::vector<KernelMemoryTraceBlockPtr> *kernelMemoryTraceBlocks,
                                       std::vector<MemoryTraceBlockPtr> *mergedMemoryTraceBlocks);

  std::shared_ptr<std::map<const DeviceContext *, std::vector<KernelMemoryTraceBlockPtr>>> kernelMemoryTraceBlocks_;
  std::shared_ptr<std::map<const DeviceContext *, std::vector<MemoryTraceBlockPtr>>> mergedMemoryTraceBlocks_;
  std::shared_ptr<std::unordered_map<OpRunner *, std::vector<KernelMemoryTraceBlockPtr>>> kernelToBlock_;
  std::shared_ptr<std::unordered_map<const Tensor *, KernelMemoryTraceBlockPtr>> tensorToKernelMemBlocks_;
};

}  // namespace runtime
}  // namespace mrt
#endif
