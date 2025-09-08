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
#ifndef INFERRT_SRC_HARDWARE_ASCEND_ASCEND_RES_MANAGER_H_
#define INFERRT_SRC_HARDWARE_ASCEND_ASCEND_RES_MANAGER_H_

#include <vector>
#include <memory>
#include <string>
#include <map>
#include <unordered_map>
#include <utility>
#include "acl/acl_rt.h"
#include "hardware/ascend/res_manager/ascend_stream_manager.h"
#include "hardware/hardware_abstract/device_event.h"
#include "hardware/hardware_abstract/device_context.h"
#include "common/visible.h"

namespace mrt {
namespace device {
namespace ascend {
std::string GetCurrentDir();

using DeviceMemInfo = std::unordered_map<device::DeviceMemPtr, std::unordered_map<std::string, size_t>>;
class MRT_EXPORT AscendResManager : public DeviceResManager {
 public:
  AscendResManager() = default;
  ~AscendResManager() override = default;

  void Initialize() override;

  void Destroy() override;

  std::shared_ptr<MemoryManager> mem_manager() const override { return memManager_; }

  std::vector<void *> AllocateContinuousMemory(const std::vector<size_t> &sizeList,
                                               uint32_t streamId = kDefaultStreamIndex) const override;
  bool IsEnableVmm() const override;

  bool BindDeviceToCurrentThread(bool forceBind) const override;
  void *GetStream() const override { return AscendStreamMng::GetInstance().default_stream(); }
  void *GetCopyDataStream() const;

  void *AllocateStaticMemory(size_t size, uint32_t streamId = kDefaultStreamIndex) const;
  void *AllocateMemory(size_t size, uint32_t streamId = kDefaultStreamIndex) const override;
  void FreeMemory(void *ptr) const override;
  void FreePartMemorys(const std::vector<void *> &freeAddrs, const std::vector<void *> &keepAddrs,
                       const std::vector<size_t> &keepAddrSizes) const override;
  void DefragMemory() override;

  size_t GetMaxUsedMemorySize() const override;

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

  bool CreateStream(size_t *streamId) const override;
  bool CreateStreamWithPriority(size_t *streamId, int32_t priority) const override;
  bool DestroyStream(size_t streamId) const override;
  size_t QueryStreamSize() const override;
  std::vector<uint32_t> GetStreamIds() const override;
  void *GetStream(size_t streamId) const override;
  void SetCurrentStreamId(size_t streamId) override;
  size_t GetCurrentStreamId() const override;
  bool QueryStream(size_t streamId) const override;
  bool SyncStream(size_t streamId = 0) const override;
  bool SyncAllStreams(bool syncDevice = true) const override;
  bool SyncNotDefaultStreams() const override;
  size_t DefaultStream() const override;

  DeviceEventPtr CreateRuntimeEvent(bool enableBlocking, bool enableRecordWait) override;
  DeviceEventPtr CreateEventWithFlag(bool enableTiming, bool blocking, bool useExtensionalApi) override;
  bool DestroyEvent(const DeviceEventPtr &event) override;
  bool DestroyAllEvents() override;

  bool singleOpMultiStreamEnable() const override;
  void set_single_op_multi_stream_enable(bool singleOpMultiStreamEnable) override;
  // Only used in graph_mode with MS_DISABLE_REF_MODE, delete it when delete MS_DISABLE_REF_MODEF
  void SetCPUMemManager();

  // Override interface for multi stream event control.
  bool RecordEvent(int64_t taskIdOnStream, uint32_t userStreamId,
                   const std::vector<std::pair<uint32_t, DeviceMemPtr>> &memoryStreamAddresses,
                   const DeviceEventPtr &inputEvent) override;

  bool WaitEvent(int64_t taskIdOnStream, uint32_t userStreamId, uint32_t memoryStreamId) override;

  bool WaitEvent(int64_t taskIdOnStream, uint32_t userStreamId) override;

  bool SyncAllEvents() override;

  bool LaunchCallback(std::function<void(void)> callbackFunc, size_t streamId, bool isBlock = false) const override;

  void ResetStreamAndCtx() const override;

 private:
  bool initialized_ = false;
  std::shared_ptr<MemoryManager> memManager_{nullptr};
  DeviceEventPtrList deviceEvents_{};
  std::mutex deviceEventsMutex_;
  uint32_t deviceId_{0};
  bool enableMemoryTracker_{false};
};
}  // namespace ascend
}  // namespace device
}  // namespace mrt
#endif  // INFERRT_SRC_HARDWARE_ASCEND_ASCEND_RES_MANAGER_H_
