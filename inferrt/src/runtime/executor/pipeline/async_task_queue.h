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

#ifndef __RUNTIME_EXECUTOR_PIPELINE_ASYNC_TASK_QUEUE_H__
#define __RUNTIME_EXECUTOR_PIPELINE_ASYNC_TASK_QUEUE_H__

#include <memory>
#include <thread>
#include <string>
#include <set>
#include <utility>
#include <functional>
#include <cstddef>

#include "hardware/hardware_abstract/device_context.h"
#include "runtime/executor/pipeline/lock_free_ring_queue.h"
#include "common/common.h"

namespace mrt {
namespace runtime {
constexpr uint64_t kLFQueueCapacity = 8192;

// AsyncTaskQueue is a lock-free asynchronous queue that supports multiple producers and a single consumer. It
// internally starts a thread to act as the consumer, allowing concurrent pushing of elements. The elements must be of a
// type that can be constructed into an std::function<void()> object.
class AsyncTaskQueue {
 public:
  explicit AsyncTaskQueue(std::string name);
  ~AsyncTaskQueue();

  void Initialize();

  // Bind device and set device context for async pipeline thread.
  void BindDevice(const std::set<const device::DeviceContext *> &device_contexts);

  // Push element to lock free queue, the args parameter must be of type std::function<void()> or convertible to this
  // type. Push is multi thread safety.
  template <typename... Args>
  void Push(Args &&...args) {
    if (!init_ || worker_ == nullptr) {
      LOG_EXCEPTION << "The queue is not initialized before.";
    }
    if (!alive_) {
      return;
    }
    if (!tasksQueue_.Push(std::forward<Args>(args)...)) {
      LOG_EXCEPTION << "Failed to push task to queue: " << name_;
    }
  }

  // Wait for all async task finish executing.
  void Wait();

  // Check the queue is empty or not.
  bool Empty() const;

  // Pause the queue, can not push element to a queue which is in pause status, AsyncTaskQueue is in pause status after
  // creation.
  void Pause();

  // Continue the queue which is in pause status.
  void Continue();

  // Thread join before the process exit.
  void WorkerJoin();

  std::thread::id GetThreadID() const;
  const std::unique_ptr<std::thread> &GetWorker() const { return worker_; }

 private:
  void WorkerLoop();
  void SetThreadName() const;
  std::unique_ptr<std::thread> worker_;
  std::string name_;

  LockFreeRingQueue<std::function<void()>, kLFQueueCapacity> tasksQueue_;
  bool init_{false};
  bool alive_{true};
};
using AsyncTaskQueuePtr = std::unique_ptr<AsyncTaskQueue>;
}  // namespace runtime
}  // namespace mrt
#endif  // __RUNTIME_EXECUTOR_PIPELINE_ASYNC_TASK_QUEUE_H__
