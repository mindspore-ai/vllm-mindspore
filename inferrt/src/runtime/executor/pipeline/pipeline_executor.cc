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

#include "runtime/executor/pipeline/pipeline_executor.h"
#include "runtime/executor/pipeline/async_task_queue_manager.h"

namespace mrt {
namespace runtime {
PipelineExecutor::PipelineExecutor(const std::shared_ptr<std::vector<OpRunner>> &opRunners,
                                   const std::map<hardware::DeviceType, device::DeviceContext *> &deviceContexts)
    : Executor(opRunners, deviceContexts), initialized_(false) {}

void PipelineExecutor::Initialize() {
  if (initialized_) {
    return;
  }
  auto &asyncTaskQueueManager = AsyncTaskQueueManager::GetInstance();
  asyncTaskQueueManager.InitializeAll();
  asyncTaskQueueManager.ContinueAll();
  asyncTaskQueueManager.BindDevice();
  asyncTaskQueueManager.PauseAll();

  initialized_ = true;
}

void PipelineExecutor::Run(bool isDynamic) {
  LOG_OUT << "Begin pipeline executor run.";
  auto &asyncTaskQueueManager = AsyncTaskQueueManager::GetInstance();
  asyncTaskQueueManager.ContinueAll();

  AsyncTaskQueue *inferQeueue = asyncTaskQueueManager.GetInferQueue();
  AsyncTaskQueue *launchQeueue = asyncTaskQueueManager.GetLaunchQueue();
  OpRunner *opRunners = opRunners_->data();
  size_t opNum = opRunners_->size();
  for (size_t i = 0; i < opNum; ++i) {
    OpRunner &opRunner = opRunners[i];

    auto inferTask = [&opRunner, launchQeueue]() {
      // Do infer shape and calculate workspace size in infer queue.
      if (auto errNo = opRunner.InferShape() != ops::SUCCESS) {
        LOG_EXCEPTION << "Infer shape failed for operator " << ops::ToStr(opRunner.GetNode()->op) << "Errno: " << errNo;
      }
      if (auto errNo = opRunner.CalcWorkspace() != ops::SUCCESS) {
        LOG_EXCEPTION << "CalcWorkspace shape failed for operator " << ops::ToStr(opRunner.GetNode()->op)
                      << "Errno: " << errNo;
      }

      // Push async launch task into launch queue.
      auto launchTask = [&opRunner]() {
        if (auto errNo = opRunner.Launch() != ops::SUCCESS) {
          LOG_EXCEPTION << "Launch shape failed for operator " << ops::ToStr(opRunner.GetNode()->op)
                        << "Errno: " << errNo;
        }
      };
      launchQeueue->Push(std::move(launchTask));
    };

    inferQeueue->Push(std::move(inferTask));
  }

  asyncTaskQueueManager.WaitAll();
  asyncTaskQueueManager.PauseAll();

  for (auto &deviceItem : deviceContexts_) {
    const auto &res_manager = deviceItem.second->deviceResManager_;
    CHECK_IF_NULL(res_manager);
    res_manager->SyncAllStreams();
  }
  LOG_OUT << "End pipeline executor run.";
}
}  // namespace runtime
}  // namespace mrt
