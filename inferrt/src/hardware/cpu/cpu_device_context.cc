/**
 * Copyright 2025-2025 Huawei Technologies Co., Ltd
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
#include <map>
#include <string>
#include <unordered_set>
#include <utility>

#include "hardware/hardware_abstract/device_context_manager.h"

namespace mrt {
namespace device {
namespace cpu {
namespace {
const char kCPUDevice[] = "CPU";

}  // namespace

void CPUDeviceContext::Initialize() {
  if (initialized_) {
    return;
  }
  deviceResManager_->Initialize();
  initialized_ = true;
}

void CPUDeviceContext::Destroy() {
  deviceResManager_->Destroy();
  initialized_ = false;
}

// Register functions to _c_expression so python hal module could call CPU device interfaces.
MS_REGISTER_DEVICE(kCPUDevice, CPUDeviceContext);
}  // namespace cpu
}  // namespace device
}  // namespace mrt
