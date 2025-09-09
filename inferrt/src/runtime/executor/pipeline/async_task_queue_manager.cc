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

#include "runtime/executor/pipeline/async_task_queue_manager.h"
#include <memory>

namespace mrt {
namespace runtime {
AsyncTaskQueueManager &AsyncTaskQueueManager::GetInstance() {
  static AsyncTaskQueueManager instance;
  return instance;
}

AsyncTaskQueueManager::AsyncTaskQueueManager()
    : inferQueue_(std::make_unique<AsyncTaskQueue>("infer_queue")),
      launchQueue_(std::make_unique<AsyncTaskQueue>("launch_queue")) {}

void AsyncTaskQueueManager::InitializeAll() {
  inferQueue_->Initialize();
  launchQueue_->Initialize();
}

void AsyncTaskQueueManager::PauseAll() {
  inferQueue_->Pause();
  launchQueue_->Pause();
}

void AsyncTaskQueueManager::ContinueAll() {
  inferQueue_->Continue();
  launchQueue_->Continue();
}

void AsyncTaskQueueManager::WaitAll() {
  inferQueue_->Wait();
  launchQueue_->Wait();
}

void AsyncTaskQueueManager::WorkerJoin() {
  inferQueue_->WorkerJoin();
  launchQueue_->WorkerJoin();
}

void AsyncTaskQueueManager::AddDeviceContext(const device::DeviceContext *deviceContext) {
  CHECK_IF_NULL(deviceContext);
  (void)deviceContexts_.insert(deviceContext);
}

const std::set<const device::DeviceContext *> &AsyncTaskQueueManager::GetAllDeviceContexts() const {
  return deviceContexts_;
}

void AsyncTaskQueueManager::BindDevice() {
  inferQueue_->BindDevice(deviceContexts_);
  launchQueue_->BindDevice(deviceContexts_);
}

}  // namespace runtime
}  // namespace mrt
