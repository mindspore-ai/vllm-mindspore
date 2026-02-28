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

#include "runtime/device_inference.h"

#include <algorithm>

#include "common/logger.h"

namespace mrt {
namespace runtime {

hardware::Device getDeviceFromOutputAndInputs(const ir::ValuePtr &output, const std::vector<ir::ValuePtr> &inputs) {
  CHECK_IF_NULL(output);

  if (output->IsTensor()) {
    auto &tensor = output->ToTensor();
    CHECK_IF_NULL(tensor);
    return tensor->GetDevice();
  }

  if (output->IsNone()) {
    auto it =
      std::find_if(inputs.begin(), inputs.end(), [](const ir::ValuePtr &v) { return v != nullptr && v->IsTensor(); });
    if (it != inputs.end()) {
      return (*it)->ToTensor()->GetDevice();
    }
    return {hardware::DeviceType::CPU, 0};
  }

  if (output->IsTuple()) {
    auto &tuple = output->ToTuple();
    CHECK_IF_NULL(tuple);

    if (tuple->Size() == 0) {
      return {hardware::DeviceType::CPU, 0};
    }

    bool allTensor = std::all_of(tuple->begin(), tuple->end(),
                                 [](const ir::ValuePtr &elem) { return elem != nullptr && elem->IsTensor(); });

    if (allTensor) {
      return (*tuple->begin())->ToTensor()->GetDevice();
    }
    return {hardware::DeviceType::CPU, 0};
  }

  return {hardware::DeviceType::CPU, 0};
}

}  // namespace runtime
}  // namespace mrt
