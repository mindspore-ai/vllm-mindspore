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

#ifndef __CONFIG_DEVICE_ASCEND_COMMON_H__
#define __CONFIG_DEVICE_ASCEND_COMMON_H__

#include <cstdint>

namespace mrt {
namespace config {
namespace ascend {
enum class AclCubeMathType : int8_t {
  KEEP_DTYPE = 0,
  ALLOW_FP32_DOWN_PRECISION = 1,
  USE_FP16 = 2,
  USE_HF32 = 3,
};

enum class SocVersion {
  UnsupportedSocVersion = -1,
  k910PremiumA = 100,
  k910ProA,
  k910A,
  k910ProB,
  k910B,
  k310P1 = 200,
  k310P2,
  k310P3,
  k310P4,
  k310P5,
  k310P7,
  k910B1 = 220,
  k910B2,
  k910B2C,
  k910B3,
  k910B4,
  k910B4_1,
  k310B1 = 240,
  k310B2,
  k310B3,
  k310B4,
  k910_9391 = 250,
  k910_9392,
  k910_9381,
  k910_9382,
  k910_9372,
  k910_9362
};

}  // namespace ascend
}  // namespace config
}  // namespace mrt
#endif  // __CONFIG_DEVICE_ASCEND_COMMON_H__
