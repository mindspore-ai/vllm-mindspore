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
  mrt::device::DeviceContextKey device_context_key{"Ascend", 0};
  auto device_context =
    mrt::device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(device_context_key);
  if (device_context == nullptr) {
    LOG_ERROR << "Get device context failed.";
  }
  if (device_context->deviceResManager_ == nullptr) {
    LOG_ERROR << "Get device res manager failed.";
  }
  device_context->Initialize();

  // Test allocate memory.
  auto ptr = device_context->deviceResManager_->AllocateMemory(8);
  LOG_ERROR << "ptr:" << ptr;

  // Test event and stream.
  size_t stream_id = 1;
  if (!device_context->deviceResManager_->CreateStream(&stream_id)) {
    LOG_ERROR << "Create stream failed.";
  }
  std::vector<std::pair<uint32_t, mrt::device::DeviceMemPtr>> memory_stream_addresses;
  memory_stream_addresses.emplace_back(0, ptr);
  auto input_event = device_context->deviceResManager_->CreateRuntimeEvent(true, true);
  int64_t task_id_on_stream = 1;
  if (!device_context->deviceResManager_->RecordEvent(task_id_on_stream, SizeToUint(stream_id),
                                                        memory_stream_addresses, input_event)) {
    LOG_ERROR << "Record event on stream failed.";
  }
  if (!device_context->deviceResManager_->WaitEvent(task_id_on_stream, SizeToUint(stream_id))) {
    LOG_ERROR << "Wait event failed.";
  }
  if (!device_context->deviceResManager_->SyncStream(0)) {
    LOG_ERROR << "Sync stream failed.";
  }

  // Free ptr and destroy event.
  device_context->deviceResManager_->FreeMemory(ptr);
  if (!device_context->deviceResManager_->DestroyAllEvents()) {
    LOG_ERROR << "Destroy event failed.";
  }
}

int main() {
  TestDeviceResource();
  return 0;
}
