/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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

#ifndef __OPS_UTILS_OP_SUPPORT_H__
#define __OPS_UTILS_OP_SUPPORT_H__

#include <cstdint>
#include <string>
#include <vector>

#include "hardware/device.h"
#include "common/visible.h"
#include "ir/common/dtype.h"
#include "ir/value/value.h"

namespace mrt {
namespace runtime {

enum class OpSupportStatus : int32_t {
  kOk = 0,
  kUnsupportedDevice = 1,
  kUnsupportedInputType = 2,
};

struct OpSupportResult {
  OpSupportStatus status{OpSupportStatus::kOk};
  std::string message;
};

MRT_EXPORT hardware::Device GetDeviceFromOutputAndInputs(const ir::ValuePtr &output,
                                                         const std::vector<ir::ValuePtr> &inputs);

MRT_EXPORT OpSupportResult CheckOpSupport(const std::string &op_name, const ir::ValuePtr &output_value,
                                          const std::vector<ir::ValuePtr> &input_values);

}  // namespace runtime
}  // namespace mrt

#endif  // __OPS_UTILS_OP_SUPPORT_H__
