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

#include "hardware/cpu/cpu_device_context.h"
#include "common/common.h"

void TestDeviceResource() {
  mrt::device::DeviceContextKey deviceContextKey{"CPU", 0};
  auto deviceContext = std::make_shared<mrt::device::cpu::CPUDeviceContext>(deviceContextKey);
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

  // Free ptr and destroy event.
  deviceContext->deviceResManager_->FreeMemory(ptr);
}

int main() {
  TestDeviceResource();
  return 0;
}
