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

#ifndef INFERRT_SRC_HARDWARE_DEVICE_CONTEXT_H_
#define INFERRT_SRC_HARDWARE_DEVICE_CONTEXT_H_

#include <map>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <mutex>
#include <functional>
#include "common/common.h"
#include "common/visible.h"
#include "hardware/hardware_abstract/stream_util.h"
#ifdef __APPLE__
#include "async/spinlock.h"
#endif

namespace mrt {
class DeviceEvent;
using DeviceEventPtr = std::shared_ptr<DeviceEvent>;
namespace runtime {
enum class KernelTaskType;
}
namespace device {
constexpr size_t kSizeZero = 0;

struct DeviceContextKey {
  // device type name, such as 'GPU' 'Ascend' 'CPU'.
  std::string deviceName_;
  uint32_t deviceId_{0};

  // Use the result of ToString() as key to look up DeviceContext
  // in cache map which maintains created DeviceContext objects.
  std::string ToString() const { return deviceName_ + "_" + std::to_string(deviceId_); }
};

class DeviceResManager;
class KernelExecutor;

// DeviceContext is unified interface of interaction with device.
class MRT_EXPORT DeviceContext {
 public:
  explicit DeviceContext(const DeviceContextKey &device_context_key)
      : deviceContextKey_(device_context_key), initialized_(false) {}
  virtual ~DeviceContext() = default;

  // Initialize the device context.
  virtual void Initialize() = 0;

  // Destroy device context and release device resource.
  virtual void Destroy() = 0;

  // Get deviceContextKey_ to obtain device name and device id.
  const DeviceContextKey &device_context_key() const { return deviceContextKey_; }

  // Get kernel executor.
  std::shared_ptr<KernelExecutor> GetKernelExecutor() const { return kernelExecutor_; }

  void SetKernelExecutor(const std::shared_ptr<KernelExecutor> &kernel_executor) { kernelExecutor_ = kernel_executor; }

  // Return whether this device context is initialized.
  bool initialized() const;

  DeviceContextKey deviceContextKey_;
  std::unique_ptr<DeviceResManager> deviceResManager_;

 protected:
#ifdef __APPLE__
  // There are some problems with using mutex on Mac, use spinlocks instead.
  inline static SpinLock initLock_;
#else
  inline static std::mutex initMutex_;
#endif
  bool initialized_;

 private:
  std::shared_ptr<KernelExecutor> kernelExecutor_;
};
using DeviceContextPtr = std::shared_ptr<DeviceContext>;
class MemoryManager;
class CollectiveCommunicationLib;
class OffloadedMemPool;
using DeviceMemPtr = void *;

class MRT_EXPORT DeviceResManager {
 public:
  DeviceResManager();

  virtual ~DeviceResManager() = default;

  // Initialize the device resource manager.
  virtual void Initialize() {}

  virtual void SetAclDeterministic() {}

  // Destroy device resource manager and release device resource.
  virtual void Destroy() {}

  // Bind device to current thread to gain device control privileges
  // If force_bind is true, bind context to current thread every time;
  // Otherwise, only bind context to current thread for the first time.
  virtual bool BindDeviceToCurrentThread(bool force_bind) const { return true; }
  virtual void ResetStreamAndCtx() const {}

  // Relevant function to allocate and free device memory of raw ptr.
  virtual void *AllocateMemory(size_t size, uint32_t stream_id = kDefaultStreamIndex) const = 0;
  virtual void FreeMemory(void *ptr) const = 0;
  virtual void FreePartMemorys(const std::vector<void *> &free_addrs, const std::vector<void *> &keep_addrs,
                               const std::vector<size_t> &keep_addr_sizes) const = 0;
  virtual void DefragMemory() {}
  virtual bool IsEnableVmm() const { return false; }

  // Interface for multi stream event control.
  virtual bool RecordEvent(int64_t task_id_on_stream, uint32_t user_stream_id,
                           const std::vector<std::pair<uint32_t, DeviceMemPtr>> &memory_stream_addresses,
                           const DeviceEventPtr &input_event) {
    return false;
  }

  virtual bool WaitEvent(int64_t task_id_on_stream, uint32_t user_stream_id, uint32_t memory_stream_id) {
    return false;
  }

  virtual bool WaitEvent(int64_t task_id_on_stream, uint32_t user_stream_id) { return false; }

  virtual bool SyncAllEvents() { return false; }

  virtual size_t GetMaxUsedMemorySize() const { return 0; }

  // Relevant function to manage memory statistics
  virtual size_t GetTotalMemStatistics() const { return 0; }
  virtual size_t GetTotalUsedMemStatistics() const { return 0; }
  virtual size_t GetTotalIdleMemStatistics() const { return 0; }
  virtual size_t GetTotalEagerFreeMemStatistics() const { return 0; }
  virtual size_t GetUsedMemPeakStatistics() const { return 0; }
  virtual size_t GetReservedMemPeakStatistics() const { return 0; }
  virtual std::unordered_map<std::string, std::size_t> GetBlockCountsStatistics() const { return {}; }
  virtual std::unordered_map<std::string, std::size_t> GetBlockUnitSizeStatistics() const { return {}; }
  virtual std::unordered_map<device::DeviceMemPtr, std::unordered_map<std::string, size_t>>
  GetCommonMemBlocksInfoStatistics() const {
    return {};
  }
  virtual std::unordered_map<device::DeviceMemPtr, std::unordered_map<std::string, size_t>>
  GetPersistentMemBlocksInfoStatistics() const {
    return {};
  }
  virtual void ResetMaxMemoryReserved() {}
  virtual void ResetMaxMemoryAllocated() {}

  virtual size_t EmptyCache() { return -1L; }

  // Allocate host memory with raii and ref count
  virtual std::shared_ptr<void> AllocateHostMemory(size_t size) const {
    return std::shared_ptr<void>(::malloc(size), ::free);
  }
  virtual size_t GetAvailableMemSize() const { return 0; }

  // Allocate continuous device memory according to size list.
  // Communication operators may need continuous memory for input and output
  // to optimize the communication performance.
  virtual std::vector<void *> AllocateContinuousMemory(const std::vector<size_t> &size_list,
                                                       uint32_t stream_id = kDefaultStreamIndex) const {
    LOG_ERROR << "Unimplemented interface.";
    return {};
  }

  // Create a stream with assigning a stream id, the assigned stream id will be written to the parameter '*stream_id'.
  virtual bool CreateStream(size_t *stream_id) const {
    LOG_ERROR << "Unimplemented interface: 'CreateStream'.";
    *stream_id = kSizeZero;
    return false;
  }

  // Create a stream with priority.
  virtual bool CreateStreamWithPriority(size_t *stream_id, int32_t priority) const {
    *stream_id = kSizeZero;
    return false;
  }

  virtual size_t QueryStreamSize() const { return 0L; }
  virtual std::vector<uint32_t> GetStreamIds() const { return {}; }

  // If multi-stream used in pynative mode, other streams must be sync before the graph
  // is executed. Otherwise, out-of-order occurs. Therefore this flag is added.
  // This solution is a temporary solution, this flag will be removed after multi-stream is
  // supported in graph mode.
  virtual bool single_op_multi_stream_enable() const { return false; }
  virtual void set_single_op_multi_stream_enable(bool single_op_multi_stream_enable) {}

  // Get the stream pointer by stream_id.
  virtual void *GetStream(size_t stream_id) const { return nullptr; }

  // Set currently using stream id.
  virtual void SetCurrentStreamId(size_t stream_id) { return; }

  // Get currently using stream id.
  virtual size_t GetCurrentStreamId() const { return kSizeZero; }

  virtual void *GetStream() const { return nullptr; }

  virtual size_t GetCommunicationStreamID() const { return kDefaultStreamIndex; }

  virtual size_t GetCommunicationStreamIDByGroup(const std::string &group) const { return GetCommunicationStreamID(); }

  // Destroy a stream bound to the input parameter "stream_id".
  virtual bool DestroyStream(size_t stream_id) const { return false; }

  // Query tasks' completion status of a stream.
  virtual bool QueryStream(size_t stream_id) const { return true; }

  // Synchronize stream, device such as GPU and Ascend need stream to launch kernel asynchronously,
  // Using 'SyncStream' to block thread and wait for completing all tasks on specific stream.
  // Using 'SyncAllStream' to block thread and wait for completing all tasks on all streams.
  // Devices without stream could ignore the implementation of these function.
  // Since the current entry for creating streams is not unified, the implementation of the 'SyncStream' and
  // "SyncAllStreams" interfaces are implemented by subclasses.
  virtual bool SyncStream(size_t stream_id) const { return true; }

  // 'sync_device' is used for Ascend backend.
  virtual bool SyncAllStreams(bool sync_device = true) const { return true; }

  virtual bool SyncNotDefaultStreams() const { return true; }

  // Return default stream id. Normally it's 0.
  virtual size_t DefaultStream() const { return 0; }

  // Create device event for runtime.
  virtual DeviceEventPtr CreateRuntimeEvent(bool enable_blocking, bool enable_record_wait) { return nullptr; }

  // Create device event with flag.
  virtual DeviceEventPtr CreateEventWithFlag(bool enable_timing, bool blocking, bool use_extensional_api = true) {
    return nullptr;
  }

  // Destroy specified device event.
  virtual bool DestroyEvent(const DeviceEventPtr &event) { return true; }

  // Destroy all device events.
  virtual bool DestroyAllEvents() { return true; }

  virtual std::shared_ptr<MemoryManager> mem_manager() const { return nullptr; }

  virtual bool LaunchCallback(std::function<void(void)> callback_func, size_t stream_id, bool is_block = false) const {
    callback_func();
    return true;
  }

 protected:
  DeviceContext *deviceContext_{nullptr};

 private:
  template <class... Args>
  friend class DeviceInterface;
  void SetDeviceContext(DeviceContext *device_context) { deviceContext_ = device_context; }
  std::shared_ptr<device::OffloadedMemPool> offloaded_mem_pool_;
};

using CallbackFunc = std::function<void(void)>;

class MRT_EXPORT KernelExecutor {
 public:
  virtual ~KernelExecutor() = default;

  virtual void Initialize() {}
  virtual void Destroy() {}

  void SetDeviceContext(DeviceContext *device_context) { deviceContext_ = device_context; }

 protected:
  DeviceContext *deviceContext_{nullptr};
};

template <class... Args>
class DeviceInterface : public DeviceContext {};

template <>
class DeviceInterface<> : public DeviceContext {
 public:
  explicit DeviceInterface(const DeviceContextKey &key) : DeviceContext(key) {}

 protected:
  void CheckUnset(const void *ptr, const std::string &error_msg) const {
    if (ptr != nullptr) {
      LOG_ERROR << error_msg;
    }
  }
};

template <class T, class... Args>
class DeviceInterface<T, Args...> : public DeviceInterface<Args...> {
 public:
  explicit DeviceInterface(const DeviceContextKey &key) : DeviceInterface<Args...>(key) {
    if constexpr (std::is_base_of_v<DeviceResManager, T>) {
      DeviceInterface::CheckUnset(reinterpret_cast<void *>(DeviceContext::deviceResManager_.get()),
                                  "DeviceResManager has been registered!");
      DeviceContext::deviceResManager_ = std::make_unique<T>();
      DeviceContext::deviceResManager_->SetDeviceContext(this);
    } else if constexpr (std::is_base_of_v<KernelExecutor, T>) {
      DeviceInterface::CheckUnset(reinterpret_cast<void *>(DeviceContext::GetKernelExecutor().get()),
                                  "KernelExecutor has been registered!");
      DeviceContext::SetKernelExecutor(std::make_shared<T>());
      DeviceContext::GetKernelExecutor()->SetDeviceContext(this);
    }
  }
};
}  // namespace device
}  // namespace mrt
#endif  // INFERRT_SRC_HARDWARE_DEVICE_CONTEXT_H_
