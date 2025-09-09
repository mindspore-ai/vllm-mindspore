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
#ifndef __RUNTIME_EXECUTOR_PIPELINE_ASYNC_TASK_QUEUE_MANAGER_H__
#define __RUNTIME_EXECUTOR_PIPELINE_ASYNC_TASK_QUEUE_MANAGER_H__

#include <set>
#include "runtime/executor/pipeline/async_task_queue.h"
#include "hardware/hardware_abstract/device_context.h"
#include "common/common.h"

namespace mrt {
namespace runtime {
// Singleton AsyncTaskQueueManager manages multi-stage asynchronous processing tasks
// Uses lock-free queues for thread-safe operations between stages: infer -> launch.
class AsyncTaskQueueManager {
 public:
  static AsyncTaskQueueManager &GetInstance();

  AsyncTaskQueue *GetInferQueue() { return inferQueue_.get(); }
  AsyncTaskQueue *GetLaunchQueue() { return launchQueue_.get(); }

  void InitializeAll();

  // Suspends all pipeline queue, can not push element to a queue which is in pause status.
  void PauseAll();
  // Continue all pipeline queue which is in pause status.
  void ContinueAll();

  // Blocks until all queued tasks complete processing.
  void WaitAll();
  // Waits for worker threads to terminate (shutdown sequence).
  void WorkerJoin();

  void AddDeviceContext(const device::DeviceContext *deviceContext);

  const std::set<const device::DeviceContext *> &GetAllDeviceContexts() const;

  // Bind device and set device context for async pipeline threads.
  void BindDevice();

 private:
  AsyncTaskQueueManager();
  ~AsyncTaskQueueManager() = default;
  DISABLE_COPY_AND_ASSIGN(AsyncTaskQueueManager);

  AsyncTaskQueuePtr inferQueue_;
  AsyncTaskQueuePtr launchQueue_;
  std::set<const device::DeviceContext *> deviceContexts_;
};
}  // namespace runtime
}  // namespace mrt
#endif  // __RUNTIME_EXECUTOR_PIPELINE_ASYNC_TASK_QUEUE_MANAGER_H__
