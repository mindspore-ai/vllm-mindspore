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
#include "hardware/cpu/res_manager/cpu_res_manager.h"
#include <utility>
#include <vector>
#include <string>
#include <unordered_map>
#include <memory>

namespace mrt {
namespace device {
namespace cpu {
void CPUResManager::Initialize() { LOG_OUT << "Unimplemented interface."; }

void CPUResManager::Destroy() { LOG_OUT << "Unimplemented interface."; }

void *CPUResManager::AllocateMemory(size_t size, uint32_t streamId) const {
  void *ptr = std::malloc(size);
  if (ptr == nullptr) {
    LOG_ERROR << "Memory allocate failed";
    return nullptr;
  }
  return ptr;
}

void CPUResManager::FreeMemory(void *ptr) const {
  CHECK_IF_NULL(ptr);
  std::free(ptr);
}

void CPUResManager::FreePartMemorys(const std::vector<void *> &freeAddrs, const std::vector<void *> &keepAddrs,
                                    const std::vector<size_t> &keepAddrSizes) const {
  LOG_OUT << "Unimplemented interface.";
  return;
}

namespace {

// clang-format off
#define FOR_EACH_TYPE_BASE(M)                    \
  M(kNumberTypeBool, bool)                       \
  M(kNumberTypeUInt8, uint8_t)                   \
  M(kNumberTypeInt4, int8_t)                     \
  M(kNumberTypeInt8, int8_t)                     \
  M(kNumberTypeInt16, int16_t)                   \
  M(kNumberTypeInt32, int32_t)                   \
  M(kNumberTypeInt64, int64_t)                   \
  M(kNumberTypeUInt16, uint16_t)                 \
  M(kNumberTypeUInt32, uint32_t)                 \
  M(kNumberTypeUInt64, uint64_t)                 \
  M(kNumberTypeFloat16, float16)                 \
  M(kNumberTypeFloat32, float)                   \
  M(kNumberTypeFloat64, double)                  \
  M(kNumberTypeFloat8E4M3FN, float8_e4m3fn)      \
  M(kNumberTypeFloat8E5M2, float8_e5m2)          \
  M(kNumberTypeHiFloat8, hifloat8)               \
  M(kNumberTypeComplex64, ComplexStorage<float>) \
  M(kNumberTypeComplex128, ComplexStorage<double>)


#define FOR_EACH_TYPE_EXTRA(M) M(kNumberTypeBFloat16, bfloat16)

#define FOR_EACH_TYPE(M) \
  FOR_EACH_TYPE_BASE(M)  \
  FOR_EACH_TYPE_EXTRA(M)

#define REGISTER_SIZE(addressTypeId, addressType) { addressTypeId, sizeof(addressType) },


#undef FOR_EACH_TYPE
#undef FOR_EACH_TYPE_BASE
#undef FOR_EACH_TYPE_EXTRA
#undef REGISTER_SIZE
// clang-format on
}  // namespace

}  // namespace cpu
}  // namespace device
}  // namespace mrt
