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
#ifndef INFERRT_SRC_HARDWARE_CPU_CPU_RES_MANAGER_H_
#define INFERRT_SRC_HARDWARE_CPU_CPU_RES_MANAGER_H_

#include <utility>
#include <vector>
#include <string>
#include <memory>
#include "hardware/hardware_abstract/device_context.h"

namespace mrt {
namespace device {
namespace cpu {
class CPUResManager : public DeviceResManager {
 public:
  CPUResManager() { Initialize(); }
  ~CPUResManager() override = default;

  void Initialize() override;

  void Destroy() override;

  // Relevant function to allocate and free device memory of raw ptr.
  void *AllocateMemory(size_t size, uint32_t streamId = kDefaultStreamIndex) const override;
  void FreeMemory(void *ptr) const override;
  void FreePartMemorys(const std::vector<void *> &freeAddrs, const std::vector<void *> &keepAddrs,
                       const std::vector<size_t> &keepAddrSizes) const override;
};
}  // namespace cpu
}  // namespace device
}  // namespace mrt
#endif
