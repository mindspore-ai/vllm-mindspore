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

#include <cstdlib>
#include <stdexcept>

#include "ir/tensor/storage.h"

namespace mrt {
namespace ir {

Storage::Storage(size_t sizeBytes, hardware::Device device) : sizeBytes_(sizeBytes), device_(device), ownsData_(true) {
  Resize(sizeBytes_);
}

Storage::Storage(void *data, size_t sizeBytes, hardware::Device device)
    : data_(data), sizeBytes_(sizeBytes), device_(device), ownsData_(false) {}

Storage::~Storage() { Resize(0); }

void Storage::Resize(size_t sizeBytes) {
  sizeBytes_ = sizeBytes;
  if (!ownsData_) {
    return;
  }
  if (device_.type == hardware::DeviceType::CPU) {
    if (data_) {
      free(data_);
    }
    if (sizeBytes_ == 0) {
      data_ = nullptr;
    } else {
      data_ = malloc(sizeBytes_);
      if (!data_) {
        throw std::bad_alloc();
      }
    }
  } else {
    // Handle other devices like GPU (e.g., cudaMalloc)
    LOG_EXCEPTION << "Device not supported yet";
  }
}

}  // namespace ir
}  // namespace mrt
