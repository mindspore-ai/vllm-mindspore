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

#ifndef INFERRT_SRC_HARDWARE_ASCEND_ASCEND_DYNAMIC_MEM_ADAPTER_H_
#define INFERRT_SRC_HARDWARE_ASCEND_ASCEND_DYNAMIC_MEM_ADAPTER_H_

#include "hardware/ascend/res_manager/mem_manager/ascend_memory_adapter.h"
#include <string>
#include <map>
#include <memory>

namespace mrt {
namespace device {
namespace ascend {
class AscendDynamicMemAdapter : public AscendMemAdapter {
 public:
  bool Initialize() override;
  bool DeInitialize() override;
  uint8_t *MallocStaticDevMem(size_t size, const std::string &tag = "") override;
  bool FreeStaticDevMem(void *addr) override;
  uint8_t *MallocDynamicDevMem(size_t size, const std::string &tag = "") override;
  void ResetDynamicMemory() override;
  std::string DevMemStatistics() const override;
  size_t GetDynamicMemUpperBound(void *min_static_addr) const override;
  [[nodiscard]] uint64_t FreeDevMemSize() const override;

 private:
  size_t hasAllocSize = 0;
  std::map<void *, std::shared_ptr<MemoryBlock>> staticMemoryBlocks_;
};
}  // namespace ascend
}  // namespace device
}  // namespace mrt
#endif  // INFERRT_SRC_HARDWARE_ASCEND_ASCEND_DYNAMIC_MEM_ADAPTER_H_
