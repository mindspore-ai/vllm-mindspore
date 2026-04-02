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
#include <memory>
#include <map>
#include <utility>
#include "runtime/executor/pipeline/async_task_queue_manager.h"
#include "ops/utils/async.h"

namespace mrt {
namespace runtime {
PipelineExecutor::PipelineExecutor(const std::shared_ptr<std::vector<OpRunner>> &opRunners,
                                   const std::map<hardware::DeviceType, device::DeviceContext *> &deviceContexts,
                                   const ir::ValuePtr &output)
    : Executor(opRunners, deviceContexts, output), initialized_(false) {}

void PipelineExecutor::Initialize() {
  if (initialized_) {
    return;
  }
  auto &asyncTaskQueueManager = AsyncTaskQueueManager::GetInstance();
  asyncTaskQueueManager.InitializeAll();
  asyncTaskQueueManager.ContinueAll();
  for (const auto &item : deviceContexts_) {
    asyncTaskQueueManager.AddDeviceContext(item.second);
  }
  asyncTaskQueueManager.BindDevice();
  asyncTaskQueueManager.PauseAll();

  initialized_ = true;
}

void PipelineExecutor::Run(bool isDynamic) {
  LOG_OUT << "Begin pipeline executor run.";

  auto &launchOpFunc = ops::OpAsync::GetLaunchOpFunc();
  if (launchOpFunc == nullptr) {
    std::ostringstream oss;
    for (auto iter = deviceContexts_.begin(); iter != deviceContexts_.end(); ++iter) {
      iter->second->deviceResManager_->BindDeviceToCurrentThread(false);
      oss << " " << hardware::GetDeviceNameByType(iter->first);
    }
    LOG_EXCEPTION << "Device" << oss.str() << " does not suppprt pipeline executor.";
  }

  auto bindTask = [this]() -> int {
    for (auto iter = deviceContexts_.begin(); iter != deviceContexts_.end(); ++iter) {
      iter->second->deviceResManager_->BindDeviceToCurrentThread(false);
      iter->second->deviceResManager_->BindCurrentStream();
    }
    return ops::SUCCESS;
  };
  launchOpFunc("bind_device", bindTask, false);

  OpRunner *opRunners = opRunners_->data();
  size_t opNum = opRunners_->size();
  for (size_t i = 0; i < opNum; ++i) {
    OpRunner &opRunner = opRunners[i];

    opRunner.UpdateTensors();
    // Do infer shape and calculate workspace size in infer queue.
    if (auto errNo = opRunner.InferShape() != ops::SUCCESS) {
      LOG_EXCEPTION << "Infer shape failed for operator " << opRunner.GetOpName() << "Errno: " << errNo;
    }
    opRunner.AllocateMemory();
    if (auto errNo = opRunner.CalcWorkspace() != ops::SUCCESS) {
      LOG_EXCEPTION << "CalcWorkspace shape failed for operator " << opRunner.GetOpName() << "Errno: " << errNo;
    }
    opRunner.AllocateWorkspaceMemory();
    opRunner.FreeMemory();

    if (!opRunner.NeedLaunch()) {
      continue;
    }

    // Push async launch task into launch queue.
    auto launchTask = [&opRunner]() -> int {
      if (auto errNo = opRunner.Launch() != ops::SUCCESS) {
        LOG_EXCEPTION << "Launch shape failed for operator " << opRunner.GetOpName() << "Errno: " << errNo;
      }
      return 0;
    };
    launchOpFunc(opRunner.GetOpName(), launchTask, false);
  }

  LOG_OUT << "End pipeline executor run.";
}
}  // namespace runtime
}  // namespace mrt
