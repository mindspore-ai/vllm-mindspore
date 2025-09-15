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

#include "runtime/executor/pipeline/async_task_queue.h"
#include "common/common.h"

namespace mrt {
namespace runtime {
constexpr size_t kThreadNameThreshold = 15;

void AsyncTaskQueue::SetThreadName() const {
  // Set thread name to monitor thread status or gdb debug.
  (void)pthread_setname_np(pthread_self(), name_.substr(0, kThreadNameThreshold).c_str());
}

AsyncTaskQueue::AsyncTaskQueue(std::string name) : name_(std::move(name)) {
  worker_ = std::make_unique<std::thread>(&AsyncTaskQueue::WorkerLoop, this);
}

AsyncTaskQueue::~AsyncTaskQueue() {
  try {
    WorkerJoin();
  } catch (const std::exception &e) {
    LOG_ERROR << "WorkerJoin failed, error msg:" << e.what();
  }
}

void AsyncTaskQueue::WorkerLoop() {
  SetThreadName();

  while (alive_) {
    auto *task = tasksQueue_.Front();
    if (task == nullptr) {
      return;
    }

    try {
      (*task)();
      tasksQueue_.Pop();
    } catch (const std::exception &e) {
      LOG_ERROR << "Run task failed and catch exception: " << e.what();
      while (!tasksQueue_.Empty()) {
        tasksQueue_.Pop();
      }
      alive_ = false;
      break;
    }
  }
}

void AsyncTaskQueue::Initialize() {
  if (init_) {
    return;
  }

  if (worker_ == nullptr) {
    worker_ = std::make_unique<std::thread>(&AsyncTaskQueue::WorkerLoop, this);
  }
  init_ = true;
}

void AsyncTaskQueue::BindDevice(const std::set<const device::DeviceContext *> &device_contexts) {
  auto bind_device_task = [&device_contexts]() {
    std::for_each(device_contexts.begin(), device_contexts.end(),
                  [](const device::DeviceContext *item) { item->deviceResManager_->BindDeviceToCurrentThread(false); });
  };
  Push(std::move(bind_device_task));
  Wait();
}

void AsyncTaskQueue::Wait() {
  if (!init_ || worker_ == nullptr) {
    return;
  }
  if (worker_->get_id() == std::this_thread::get_id()) {
    return;
  }

  std::atomic<bool> atomicWaitFlag = false;
  auto waitTask = [&atomicWaitFlag]() { atomicWaitFlag = true; };
  Push(std::move(waitTask));

  while (atomicWaitFlag == false) {
  }
}

bool AsyncTaskQueue::Empty() const { return tasksQueue_.Empty(); }

void AsyncTaskQueue::Pause() {
  if (!init_) {
    return;
  }

  if (tasksQueue_.IsPaused()) {
    // Has beed paused already.
    return;
  }

  Wait();
  tasksQueue_.Pause();
}

void AsyncTaskQueue::Continue() {
  if (!init_) {
    return;
  }
  tasksQueue_.Continue();
}

std::thread::id AsyncTaskQueue::GetThreadID() const {
  CHECK_IF_NULL(worker_);
  return worker_->get_id();
}

void AsyncTaskQueue::WorkerJoin() {
  if (worker_ == nullptr) {
    return;
  }
  if (init_) {
    while (!Empty()) {
    }
  }

  alive_ = false;
  tasksQueue_.Finalize();

  if (worker_->joinable()) {
    worker_->join();
  }
  worker_ = nullptr;
}
}  // namespace runtime
}  // namespace mrt
