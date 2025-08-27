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
#ifndef INFERRT_SRC_HARDWARE_ASCEND_ASCEND_DEVICE_CONTEXT_H_
#define INFERRT_SRC_HARDWARE_ASCEND_ASCEND_DEVICE_CONTEXT_H_

#include <memory>
#include <string>
#include <map>
#include "common/common.h"
#include "hardware/hardware_abstract/device_context.h"
#include "hardware/hardware_abstract/memory_manager.h"
#include "hardware/ascend/res_manager/ascend_res_manager.h"

namespace mrt {
namespace device {
namespace ascend {
class AscendResManager;

class AscendDeviceContext : public DeviceInterface<AscendResManager> {
 public:
  explicit AscendDeviceContext(const DeviceContextKey &device_context_key) : DeviceInterface(device_context_key) {}
  ~AscendDeviceContext() override = default;

  void Initialize() override;

  void InitializeForAclop() const;

  void Destroy() override;

 private:
  DISABLE_COPY_AND_ASSIGN(AscendDeviceContext);

  mutable bool initializedAclop_{false};
  pid_t pid_;  // Indicates the process id which creates the context.
};
}  // namespace ascend
}  // namespace device
}  // namespace mrt

#endif  // INFERRT_SRC_HARDWARE_ASCEND_ASCEND_DEVICE_CONTEXT_H_
