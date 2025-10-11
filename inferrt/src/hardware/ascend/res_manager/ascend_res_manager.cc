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

#include "hardware/ascend/res_manager/ascend_res_manager.h"
#ifndef _WIN32
#include <dlfcn.h>
#include <libgen.h>
#endif
#include <utility>
#include <unordered_set>
#include <vector>
#include <algorithm>
#include <numeric>
#include <set>
#include <mutex>
#include "hardware/hardware_abstract/dlopen_macro.h"
#include "hardware/ascend/res_manager/mem_manager/ascend_memory_manager.h"
#include "hardware/ascend/res_manager/mem_manager/ascend_vmm_adapter.h"
#include "hardware/ascend/res_manager/ascend_event.h"
#include "hardware/ascend/res_manager/symbol_interface/acl_compiler_symbol.h"
#include "hardware/ascend/res_manager/symbol_interface/acl_rt_symbol.h"
#include "hardware/ascend/res_manager/symbol_interface/symbol_utils.h"
#include "acl/acl_rt.h"
#include "hardware/hardware_abstract/device_context.h"
#include "hardware/hardware_abstract/device_context_manager.h"
#include "hardware/ascend/res_manager/ascend_hal_manager.h"
#include "common/common.h"

#include "hardware/hardware_abstract/collective/collective_manager.h"

namespace mrt {
namespace device {
namespace ascend {
namespace {
constexpr uint32_t kDefaultHcclExecTimeout = 1800;

using Callback = std::function<void(void)>;
std::mutex set_opt_mutex;

void AclrtLaunchCallback(void *userData) {
  Callback *callbackFunc = reinterpret_cast<Callback *>(userData);
  (*callbackFunc)();
  delete callbackFunc;
}
}  // namespace

void AscendResManager::Initialize() {
  deviceId_ = mrt::collective::CollectiveManager::Instance().local_rank_id();
  if (initialized_) {
    AscendHalManager::GetInstance().SetContextForce(deviceId_);
    return;
  }
  // init device
  AscendHalManager::GetInstance().InitDevice(deviceId_);
  AscendStreamMng::GetInstance().CreateDefaultStream();
  memManager_ = std::make_shared<EnhancedAscendMemoryManager>();
  CHECK_IF_NULL(memManager_);
  memManager_->Initialize();
  initialized_ = true;
}

void AscendResManager::Destroy() {
  if (!initialized_) {
    AscendHalManager::GetInstance().SetContextForce(deviceId_);
    return;
  }
  // To avoid call aclrtProcessReport after process exit, we should to clear all callback threads first.
  AscendStreamMng::GetInstance().Clear();

  (void)DestroyAllEvents();

  AscendStreamMng::GetInstance().DestroyAllRtEvents();
  if (!AscendStreamMng::GetInstance().DestroyAllStreams()) {
    LOG_ERROR << "Fail to destroy all streams when reset device.";
  }
  // Release memory.
  if (memManager_ != nullptr) {
    memManager_->Finalize();
    memManager_ = nullptr;
  }

  // All unmap/free operations will fail after calling aclrtResetDevice in ResetDevice,
  // so it must be called before that.
  AscendVmmAdapter::GetInstance().ClearAllMemory();
  AscendHalManager::GetInstance().ResetDevice(deviceId_);

  initialized_ = false;
}

bool AscendResManager::IsEnableVmm() const { return AscendVmmAdapter::GetInstance().IsEnabled(); }

void *AscendResManager::AllocateMemory(size_t size, uint32_t streamId) const {
  AscendHalManager::GetInstance().SetContext(deviceId_);
  CHECK_IF_NULL(memManager_);
  return memManager_->MallocMemFromMemPool(size, false, false, streamId);
}

void *AscendResManager::AllocateStaticMemory(size_t size, uint32_t streamId) const {
  AscendHalManager::GetInstance().SetContext(deviceId_);
  return memManager_->MallocMemFromMemPool(size, true, false, streamId);
}

size_t AscendResManager::GetMaxUsedMemorySize() const {
  CHECK_IF_NULL(memManager_);
  return memManager_->GetMaxUsedMemorySize();
}

void AscendResManager::FreeMemory(void *ptr) const {
  CHECK_IF_NULL(ptr);
  CHECK_IF_NULL(memManager_);
  memManager_->FreeMemFromMemPool(ptr);
}

void AscendResManager::FreePartMemorys(const std::vector<void *> &freeAddrs, const std::vector<void *> &keepAddrs,
                                       const std::vector<size_t> &keepAddrSizes) const {
  AscendMemoryPool::GetInstance().FreePartTensorMems(freeAddrs, keepAddrs, keepAddrSizes);
}

void AscendResManager::DefragMemory() { AscendMemoryPool::GetInstance().DefragMemory(); }

// Relevant function to manage memory statistics
size_t AscendResManager::GetTotalMemStatistics() const {
  CHECK_IF_NULL(memManager_);
  return memManager_->GetTotalMemStatistics();
}

size_t AscendResManager::GetTotalUsedMemStatistics() const {
  CHECK_IF_NULL(memManager_);
  return memManager_->GetTotalUsedMemStatistics();
}

size_t AscendResManager::GetTotalIdleMemStatistics() const {
  CHECK_IF_NULL(memManager_);
  return memManager_->GetTotalIdleMemStatistics();
}

size_t AscendResManager::GetTotalEagerFreeMemStatistics() const {
  CHECK_IF_NULL(memManager_);
  return memManager_->GetTotalEagerFreeMemStatistics();
}

size_t AscendResManager::GetUsedMemPeakStatistics() const {
  CHECK_IF_NULL(memManager_);
  return memManager_->GetUsedMemPeakStatistics();
}

size_t AscendResManager::GetReservedMemPeakStatistics() const {
  CHECK_IF_NULL(memManager_);
  return memManager_->GetReservedMemPeakStatistics();
}

std::unordered_map<std::string, std::size_t> AscendResManager::GetBlockCountsStatistics() const {
  CHECK_IF_NULL(memManager_);
  return memManager_->GetBlockCountsStatistics();
}

std::unordered_map<std::string, std::size_t> AscendResManager::GetBlockUnitSizeStatistics() const {
  CHECK_IF_NULL(memManager_);
  return memManager_->GetBlockUnitSizeStatistics();
}

DeviceMemInfo AscendResManager::GetCommonMemBlocksInfoStatistics() const {
  CHECK_IF_NULL(memManager_);
  return memManager_->GetCommonMemBlocksInfoStatistics();
}

DeviceMemInfo AscendResManager::GetPersistentMemBlocksInfoStatistics() const {
  CHECK_IF_NULL(memManager_);
  return memManager_->GetPersistentMemBlocksInfoStatistics();
}

void AscendResManager::ResetMaxMemoryReserved() {
  CHECK_IF_NULL(memManager_);
  auto memory_pool = memManager_->GetMemoryPool();
  CHECK_IF_NULL(memory_pool);
  memory_pool->ResetMaxMemReserved();
}

void AscendResManager::ResetMaxMemoryAllocated() {
  CHECK_IF_NULL(memManager_);
  auto memory_pool = memManager_->GetMemoryPool();
  CHECK_IF_NULL(memory_pool);
  memory_pool->ResetMaxMemAllocated();
}

size_t AscendResManager::EmptyCache() {
  CHECK_IF_NULL(memManager_);
  auto memory_pool = memManager_->GetMemoryPool();
  CHECK_IF_NULL(memory_pool);
  return memory_pool->EmptyCache();
}

std::vector<void *> AscendResManager::AllocateContinuousMemory(const std::vector<size_t> &sizeList,
                                                               uint32_t streamId) const {
  AscendHalManager::GetInstance().SetContext(deviceId_);

  CHECK_IF_NULL(memManager_);
  std::vector<size_t> alignedSizeList;
  for (auto size : sizeList) {
    auto alignSize = device::MemoryManager::GetCommonAlignSize(size);
    alignedSizeList.emplace_back(alignSize);
  }
  return memManager_->MallocContinuousMemFromMemPool(alignedSizeList, streamId);
}

bool AscendResManager::BindDeviceToCurrentThread(bool forceBind) const {
  static thread_local std::once_flag isSet;
  std::call_once(isSet, [this]() {
    auto ret = CALL_ASCEND_API(aclrtSetDevice, static_cast<int32_t>(deviceId_));
    if (ret != ACL_SUCCESS) {
      LOG_ERROR << "Device " << deviceId_ << " call aclrtSetDevice failed, ret:" << static_cast<int>(ret);
    }
  });

  if (forceBind) {
    AscendHalManager::GetInstance().SetContextForce(deviceId_);
  } else {
    AscendHalManager::GetInstance().SetContext(deviceId_);
  }

  return true;
}

bool AscendResManager::CreateStream(size_t *streamId) const {
  if (!BindDeviceToCurrentThread(false)) {
    LOG_ERROR << "Bind context to current thread failed";
    return false;
  }
  AscendStreamMng::GetInstance().CreateStream(streamId);
  return true;
}

bool AscendResManager::CreateStreamWithPriority(size_t *streamId, int32_t priority) const {
  if (!BindDeviceToCurrentThread(false)) {
    LOG_ERROR << "Bind context to current thread failed";
    return false;
  }
  AscendStreamMng::GetInstance().CreateStreamWithFlags(streamId, ACL_STREAM_FAST_LAUNCH | ACL_STREAM_FAST_SYNC,
                                                       static_cast<uint32_t>(priority));
  return true;
}

bool AscendResManager::DestroyStream(size_t streamId) const {
  if (!BindDeviceToCurrentThread(false)) {
    LOG_ERROR << "Bind context to current thread failed";
    return false;
  }
  AscendStreamMng::GetInstance().DestroyStream(streamId);
  return true;
}

size_t AscendResManager::QueryStreamSize() const { return AscendStreamMng::GetInstance().QueryStreamSize(); }

std::vector<uint32_t> AscendResManager::GetStreamIds() const { return AscendStreamMng::GetInstance().GetStreamIds(); }

bool AscendResManager::singleOpMultiStreamEnable() const {
  return AscendStreamMng::GetInstance().singleOpMultiStreamEnable();
}

void AscendResManager::set_single_op_multi_stream_enable(bool singleOpMultiStreamEnable) {
  return AscendStreamMng::GetInstance().set_single_op_multi_stream_enable(singleOpMultiStreamEnable);
}

void *AscendResManager::GetStream(size_t streamId) const {
  if (!BindDeviceToCurrentThread(false)) {
    LOG_ERROR << "Bind context to current thread failed";
    return nullptr;
  }
  return AscendStreamMng::GetInstance().GetStream(streamId);
}

void AscendResManager::SetCurrentStreamId(size_t streamId) {
  if (!BindDeviceToCurrentThread(false)) {
    LOG_ERROR << "Bind context to current thread failed";
    return;
  }
  AscendStreamMng::GetInstance().set_current_stream(streamId);
}

size_t AscendResManager::GetCurrentStreamId() const {
  if (!BindDeviceToCurrentThread(false)) {
    LOG_ERROR << "Bind context to current thread failed";
    return SIZE_MAX;
  }
  return AscendStreamMng::GetInstance().current_stream();
}

bool AscendResManager::QueryStream(size_t streamId) const {
  if (!BindDeviceToCurrentThread(false)) {
    LOG_ERROR << "Bind context to current thread failed";
    return false;
  }
  return AscendStreamMng::GetInstance().QueryStream(streamId);
}

bool AscendResManager::SyncStream(size_t streamId) const {
  if (!BindDeviceToCurrentThread(false)) {
    LOG_ERROR << "Bind context to current thread failed";
    return false;
  }
  return AscendStreamMng::GetInstance().SyncStream(streamId);
}

bool AscendResManager::SyncAllStreams(bool syncDevice) const {
  AscendHalManager::GetInstance().SetContext(deviceId_);
  return AscendStreamMng::GetInstance().SyncAllStreams(syncDevice);
}

bool AscendResManager::SyncNotDefaultStreams() const {
  if (!BindDeviceToCurrentThread(false)) {
    LOG_ERROR << "Bind context to current thread failed";
    return false;
  }
  return AscendStreamMng::GetInstance().SyncNotDefaultStreams();
}

size_t AscendResManager::DefaultStream() const {
  if (!BindDeviceToCurrentThread(false)) {
    LOG_ERROR << "Bind context to current thread failed";
    return SIZE_MAX;
  }
  return AscendStreamMng::GetInstance().default_stream_id();
}

// ACL_EVENT_TIME_LINE: indicates that the number of created events is not limited, and the created events can be used
//  to compute the elapsed time between events, which may cause lost some performance.
// ACL_EVENT_SYNC: indicates that the number of created events is limited, and the created events can be used for
//  synchronization between multiple streams.
// ACL_EVENT_CAPTURE_STREAM_PROGRESS: indicates that the number of created events is not limited and high performance,
//  and the created events can not be used for timing and synchronization.
DeviceEventPtr AscendResManager::CreateRuntimeEvent(bool enableBlocking, bool enableRecordWait) {
  if (!enableBlocking && !enableRecordWait) {
    LOG_ERROR << "Bad parameters, enableBlocking is false and enableRecordWait is false.";
  }

  uint32_t flag = 0;
  if (enableBlocking) {
    flag |= ACL_EVENT_SYNC;
  }
  if (enableRecordWait) {
    flag |= ACL_EVENT_CAPTURE_STREAM_PROGRESS;
  }
  return std::make_shared<AscendEvent>(flag);
}

DeviceEventPtr AscendResManager::CreateEventWithFlag(bool enableTiming, bool blocking, bool useExtensionalApi) {
  auto flag = enableTiming ? (ACL_EVENT_TIME_LINE | ACL_EVENT_SYNC) : ACL_EVENT_SYNC;
  auto event = std::make_shared<AscendEvent>(flag, useExtensionalApi);
  CHECK_IF_NULL(event);
  std::lock_guard<std::mutex> lock(deviceEventsMutex_);
  deviceEvents_.push_back(event);
  return event;
}

bool AscendResManager::DestroyEvent(const DeviceEventPtr &event) {
  CHECK_IF_NULL(event);
  if (!event->DestroyEvent()) {
    LOG_ERROR << "Destroy Event failed.";
    return false;
  }
  std::lock_guard<std::mutex> lock(deviceEventsMutex_);
  const auto &iter = std::find(deviceEvents_.begin(), deviceEvents_.end(), event);
  if (iter == deviceEvents_.end()) {
    LOG_OUT << "Can't find specified device event.";
    return false;
  }
  (void)deviceEvents_.erase(iter);
  return true;
}

bool AscendResManager::DestroyAllEvents() {
  DeviceEventPtrList device_events_inner;
  {
    std::lock_guard<std::mutex> lock(deviceEventsMutex_);
    device_events_inner = deviceEvents_;
    deviceEvents_.clear();
  }
  (void)std::for_each(device_events_inner.begin(), device_events_inner.end(), [this](const auto &event) {
    CHECK_IF_NULL(event);
    if (!event->DestroyEvent()) {
      LOG_ERROR << "Destroy Event failed.";
    }
  });
  deviceEvents_.clear();
  return true;
}

void *AscendResManager::GetCopyDataStream() const {
  auto copyOutDataStream = AscendStreamMng::GetInstance().GetCopyOutStream();
  if (copyOutDataStream == nullptr) {
    size_t copyStreamId;
    AscendStreamMng::GetInstance().CreateStream(&copyStreamId);
    LOG_OUT << "Create ascend copy data stream, stream id: " << copyStreamId;
    copyOutDataStream = AscendStreamMng::GetInstance().GetStream(copyStreamId);
    AscendStreamMng::GetInstance().SetCopyOutStream(copyOutDataStream);
  }
  return copyOutDataStream;
}

bool AscendResManager::RecordEvent(int64_t taskIdOnStream, uint32_t userStreamId,
                                   const std::vector<std::pair<uint32_t, DeviceMemPtr>> &memoryStreamAddresses,
                                   const DeviceEventPtr &inputEvent) {
  return memManager_->RecordEvent(taskIdOnStream, userStreamId, memoryStreamAddresses, inputEvent);
}

bool AscendResManager::WaitEvent(int64_t taskIdOnStream, uint32_t userStreamId, uint32_t memoryStreamId) {
  return memManager_->WaitEvent(taskIdOnStream, userStreamId, memoryStreamId);
}

bool AscendResManager::WaitEvent(int64_t taskIdOnStream, uint32_t userStreamId) {
  return memManager_->WaitEvent(taskIdOnStream, userStreamId);
}

bool AscendResManager::SyncAllEvents() { return memManager_->SyncAllEvents(); }

bool AscendResManager::LaunchCallback(std::function<void(void)> callbackFunc, size_t streamId, bool isBlock) const {
  auto stream = AscendStreamMng::GetInstance().GetStream(streamId);
  if (stream == nullptr) {
    stream = AscendStreamMng::GetInstance().default_stream();
  }
  CHECK_IF_NULL(stream);
  auto blockType = isBlock ? aclrtCallbackBlockType::ACL_CALLBACK_BLOCK : aclrtCallbackBlockType::ACL_CALLBACK_NO_BLOCK;
  auto callbackFuncPtr = new Callback(callbackFunc);
  aclError ret = CALL_ASCEND_API(aclrtLaunchCallback, AclrtLaunchCallback, callbackFuncPtr, blockType, stream);
  LOG_OUT << "Launch callback for streamId : " << streamId << ", ret : " << ret << ".";
  if (ret) {
    delete callbackFuncPtr;
    LOG_ERROR << "Launch callback for streamId : " << streamId << " failed, ret : " << ret << ".";
    if (SyncStream(streamId)) {
      callbackFunc();
      return true;
    }

    ResetStreamAndCtx();
    return false;
  }
  return true;
}

void AscendResManager::ResetStreamAndCtx() const {
  AscendStreamMng::GetInstance().DestroyAllStreams();
  AscendHalManager::GetInstance().ResetContext(deviceId_);
  AscendStreamMng::GetInstance().CreateDefaultStream();
}

bool AscendResManager::MemcpyDeviceToDevice(void *dst, size_t dst_size, const void *src, size_t src_size,
                                            aclrtStream stream) {
  auto ret = CALL_ASCEND_API(aclrtMemcpyAsync, dst, dst_size, src, src_size, ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
  if (ret != ACL_SUCCESS) {
    LOG_ERROR << " call aclrtMemcpyAsync failed, ret:" << static_cast<int>(ret);
    return false;
  }
  return true;
}

bool AscendResManager::MemcpyDeviceToHost(void *dst, size_t dst_size, const void *src, size_t src_size,
                                          aclrtStream stream) {
  auto ret = CALL_ASCEND_API(aclrtMemcpyAsync, dst, dst_size, src, src_size, ACL_MEMCPY_DEVICE_TO_HOST, stream);
  if (ret != ACL_SUCCESS) {
    LOG_ERROR << " call aclrtMemcpyAsync failed, ret:" << static_cast<int>(ret);
    return false;
  }
  return true;
}

}  // namespace ascend
}  // namespace device
}  // namespace mrt
