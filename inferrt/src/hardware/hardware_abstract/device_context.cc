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

#include "hardware/hardware_abstract/device_context.h"

namespace mrt {
namespace device {
DeviceResManager::DeviceResManager() { deviceContext_ = nullptr; }

bool DeviceContext::initialized() const { return initialized_; }

DeviceContextKey DeviceToDeviceContextKey(hardware::Device device) {
  uint32_t deviceId = static_cast<uint32_t>(std::max(static_cast<int32_t>(0), static_cast<int32_t>(device.index)));
  return {hardware::GetDeviceNameByType(device.type), deviceId};
}
}  // namespace device
}  // namespace mrt
