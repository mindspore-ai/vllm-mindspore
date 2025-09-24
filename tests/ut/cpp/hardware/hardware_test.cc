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

#include "hardware/ascend/ascend_device_context.h"
#include "hardware/hardware_abstract/device_context_manager.h"
#include "common/common.h"

void TestDeviceResource() {
  mrt::device::DeviceContextKey deviceContextKey{"Ascend", 0};
  auto deviceContext =
    mrt::device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(deviceContextKey);
  if (deviceContext == nullptr) {
    LOG_ERROR << "Get device context failed.";
  }
  if (deviceContext->deviceResManager_ == nullptr) {
    LOG_ERROR << "Get device res manager failed.";
  }
  deviceContext->Initialize();

  // Test allocate memory.
  auto ptr = deviceContext->deviceResManager_->AllocateMemory(8);
  LOG_ERROR << "ptr:" << ptr;

  // Test event and stream.
  size_t streamId = 1;
  if (!deviceContext->deviceResManager_->CreateStream(&streamId)) {
    LOG_ERROR << "Create stream failed.";
  }
  std::vector<std::pair<uint32_t, mrt::device::DeviceMemPtr>> memoryStreamAddresses;
  memoryStreamAddresses.emplace_back(0, ptr);
  auto inputEvent = deviceContext->deviceResManager_->CreateRuntimeEvent(true, true);
  int64_t taskIdOnStream = 1;
  if (!deviceContext->deviceResManager_->RecordEvent(taskIdOnStream, SizeToUint(streamId),
                                                        memoryStreamAddresses, inputEvent)) {
    LOG_ERROR << "Record event on stream failed.";
  }
  if (!deviceContext->deviceResManager_->WaitEvent(taskIdOnStream, SizeToUint(streamId))) {
    LOG_ERROR << "Wait event failed.";
  }
  if (!deviceContext->deviceResManager_->SyncStream(0)) {
    LOG_ERROR << "Sync stream failed.";
  }

  // Free ptr and destroy event.
  deviceContext->deviceResManager_->FreeMemory(ptr);
  if (!deviceContext->deviceResManager_->DestroyAllEvents()) {
    LOG_ERROR << "Destroy event failed.";
  }
}

int main() {
  TestDeviceResource();
  return 0;
}
