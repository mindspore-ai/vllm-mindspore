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

namespace mrt {
namespace device {
namespace ascend {
namespace {
constexpr uint32_t kDefaultHcclExecTimeout = 1800;

using Callback = std::function<void(void)>;
std::mutex set_opt_mutex;

void AclrtLaunchCallback(void *user_data) {
  Callback *callback_func = reinterpret_cast<Callback *>(user_data);
  (*callback_func)();
  delete callback_func;
}
}  // namespace

void AscendResManager::Initialize() {
  // use 0 temporarily.
  deviceId_ = 0;
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

void *AscendResManager::AllocateMemory(size_t size, uint32_t stream_id) const {
  AscendHalManager::GetInstance().SetContext(deviceId_);
  CHECK_IF_NULL(memManager_);
  return memManager_->MallocMemFromMemPool(size, false, false, stream_id);
}

void *AscendResManager::AllocateStaticMemory(size_t size, uint32_t stream_id) const {
  AscendHalManager::GetInstance().SetContext(deviceId_);
  return memManager_->MallocMemFromMemPool(size, true, false, stream_id);
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

void AscendResManager::FreePartMemorys(const std::vector<void *> &free_addrs, const std::vector<void *> &keep_addrs,
                                       const std::vector<size_t> &keep_addr_sizes) const {
  AscendMemoryPool::GetInstance().FreePartTensorMems(free_addrs, keep_addrs, keep_addr_sizes);
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

std::vector<void *> AscendResManager::AllocateContinuousMemory(const std::vector<size_t> &size_list,
                                                               uint32_t stream_id) const {
  AscendHalManager::GetInstance().SetContext(deviceId_);

  CHECK_IF_NULL(memManager_);
  std::vector<size_t> aligned_size_list;
  for (auto size : size_list) {
    auto align_size = device::MemoryManager::GetCommonAlignSize(size);
    aligned_size_list.emplace_back(align_size);
  }
  return memManager_->MallocContinuousMemFromMemPool(aligned_size_list, stream_id);
}

bool AscendResManager::BindDeviceToCurrentThread(bool force_bind) const {
  static thread_local std::once_flag is_set;
  std::call_once(is_set, [this]() {
    auto ret = CALL_ASCEND_API(aclrtSetDevice, static_cast<int32_t>(deviceId_));
    if (ret != ACL_SUCCESS) {
      LOG_ERROR << "Device " << deviceId_ << " call aclrtSetDevice failed, ret:" << static_cast<int>(ret);
    }
  });

  if (force_bind) {
    AscendHalManager::GetInstance().SetContextForce(deviceId_);
  } else {
    AscendHalManager::GetInstance().SetContext(deviceId_);
  }

  return true;
}

bool AscendResManager::CreateStream(size_t *stream_id) const {
  if (!BindDeviceToCurrentThread(false)) {
    LOG_ERROR << "Bind context to current thread failed";
    return false;
  }
  AscendStreamMng::GetInstance().CreateStream(stream_id);
  return true;
}

bool AscendResManager::CreateStreamWithPriority(size_t *stream_id, int32_t priority) const {
  if (!BindDeviceToCurrentThread(false)) {
    LOG_ERROR << "Bind context to current thread failed";
    return false;
  }
  AscendStreamMng::GetInstance().CreateStreamWithFlags(stream_id, ACL_STREAM_FAST_LAUNCH | ACL_STREAM_FAST_SYNC,
                                                       static_cast<uint32_t>(priority));
  return true;
}

bool AscendResManager::DestroyStream(size_t stream_id) const {
  if (!BindDeviceToCurrentThread(false)) {
    LOG_ERROR << "Bind context to current thread failed";
    return false;
  }
  AscendStreamMng::GetInstance().DestroyStream(stream_id);
  return true;
}

size_t AscendResManager::QueryStreamSize() const { return AscendStreamMng::GetInstance().QueryStreamSize(); }

std::vector<uint32_t> AscendResManager::GetStreamIds() const { return AscendStreamMng::GetInstance().GetStreamIds(); }

bool AscendResManager::single_op_multi_stream_enable() const {
  return AscendStreamMng::GetInstance().single_op_multi_stream_enable();
}

void AscendResManager::set_single_op_multi_stream_enable(bool single_op_multi_stream_enable) {
  return AscendStreamMng::GetInstance().set_single_op_multi_stream_enable(single_op_multi_stream_enable);
}

void *AscendResManager::GetStream(size_t stream_id) const {
  if (!BindDeviceToCurrentThread(false)) {
    LOG_ERROR << "Bind context to current thread failed";
    return nullptr;
  }
  return AscendStreamMng::GetInstance().GetStream(stream_id);
}

void AscendResManager::SetCurrentStreamId(size_t stream_id) {
  if (!BindDeviceToCurrentThread(false)) {
    LOG_ERROR << "Bind context to current thread failed";
    return;
  }
  AscendStreamMng::GetInstance().set_current_stream(stream_id);
}

size_t AscendResManager::GetCurrentStreamId() const {
  if (!BindDeviceToCurrentThread(false)) {
    LOG_ERROR << "Bind context to current thread failed";
    return SIZE_MAX;
  }
  return AscendStreamMng::GetInstance().current_stream();
}

bool AscendResManager::QueryStream(size_t stream_id) const {
  if (!BindDeviceToCurrentThread(false)) {
    LOG_ERROR << "Bind context to current thread failed";
    return false;
  }
  return AscendStreamMng::GetInstance().QueryStream(stream_id);
}

bool AscendResManager::SyncStream(size_t stream_id) const {
  if (!BindDeviceToCurrentThread(false)) {
    LOG_ERROR << "Bind context to current thread failed";
    return false;
  }
  return AscendStreamMng::GetInstance().SyncStream(stream_id);
}

bool AscendResManager::SyncAllStreams(bool sync_device) const {
  AscendHalManager::GetInstance().SetContext(deviceId_);
  return AscendStreamMng::GetInstance().SyncAllStreams(sync_device);
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
DeviceEventPtr AscendResManager::CreateRuntimeEvent(bool enable_blocking, bool enable_record_wait) {
  if (!enable_blocking && !enable_record_wait) {
    LOG_ERROR << "Bad parameters, enable_blocking is false and enable_record_wait is false.";
  }

  uint32_t flag = 0;
  if (enable_blocking) {
    flag |= ACL_EVENT_SYNC;
  }
  if (enable_record_wait) {
    flag |= ACL_EVENT_CAPTURE_STREAM_PROGRESS;
  }
  return std::make_shared<AscendEvent>(flag);
}

DeviceEventPtr AscendResManager::CreateEventWithFlag(bool enable_timing, bool blocking, bool use_extensional_api) {
  auto flag = enable_timing ? (ACL_EVENT_TIME_LINE | ACL_EVENT_SYNC) : ACL_EVENT_SYNC;
  auto event = std::make_shared<AscendEvent>(flag, use_extensional_api);
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
  auto copy_out_data_stream = AscendStreamMng::GetInstance().GetCopyOutStream();
  if (copy_out_data_stream == nullptr) {
    size_t copy_stream_id;
    AscendStreamMng::GetInstance().CreateStream(&copy_stream_id);
    LOG_OUT << "Create ascend copy data stream, stream id: " << copy_stream_id;
    copy_out_data_stream = AscendStreamMng::GetInstance().GetStream(copy_stream_id);
    AscendStreamMng::GetInstance().SetCopyOutStream(copy_out_data_stream);
  }
  return copy_out_data_stream;
}

bool AscendResManager::RecordEvent(int64_t task_id_on_stream, uint32_t user_stream_id,
                                   const std::vector<std::pair<uint32_t, DeviceMemPtr>> &memory_stream_addresses,
                                   const DeviceEventPtr &input_event) {
  return memManager_->RecordEvent(task_id_on_stream, user_stream_id, memory_stream_addresses, input_event);
}

bool AscendResManager::WaitEvent(int64_t task_id_on_stream, uint32_t user_stream_id, uint32_t memory_stream_id) {
  return memManager_->WaitEvent(task_id_on_stream, user_stream_id, memory_stream_id);
}

bool AscendResManager::WaitEvent(int64_t task_id_on_stream, uint32_t user_stream_id) {
  return memManager_->WaitEvent(task_id_on_stream, user_stream_id);
}

bool AscendResManager::SyncAllEvents() { return memManager_->SyncAllEvents(); }

bool AscendResManager::LaunchCallback(std::function<void(void)> callback_func, size_t stream_id, bool is_block) const {
  auto stream = AscendStreamMng::GetInstance().GetStream(stream_id);
  if (stream == nullptr) {
    stream = AscendStreamMng::GetInstance().default_stream();
  }
  CHECK_IF_NULL(stream);
  auto block_type =
    is_block ? aclrtCallbackBlockType::ACL_CALLBACK_BLOCK : aclrtCallbackBlockType::ACL_CALLBACK_NO_BLOCK;
  auto callback_func_ptr = new Callback(callback_func);
  aclError ret = CALL_ASCEND_API(aclrtLaunchCallback, AclrtLaunchCallback, callback_func_ptr, block_type, stream);
  LOG_OUT << "Launch callback for stream_id : " << stream_id << ", ret : " << ret << ".";
  if (ret) {
    delete callback_func_ptr;
    LOG_ERROR << "Launch callback for stream_id : " << stream_id << " failed, ret : " << ret << ".";
    if (SyncStream(stream_id)) {
      callback_func();
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

}  // namespace ascend
}  // namespace device
}  // namespace mrt
