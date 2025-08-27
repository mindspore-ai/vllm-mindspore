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

/**
 * @brief Constructs a Storage, allocating memory on the specified device.
 * @param sizeBytes The size of the storage in bytes.
 * @param device The device to allocate memory on.
 * @throws std::bad_alloc if memory allocation fails.
 * @throws std::runtime_error if the device type is not supported.
 */
Storage::Storage(size_t sizeBytes, hardware::Device device) : sizeBytes_(sizeBytes), device_(device), ownsData_(true) {
  if (device.type == hardware::DeviceType::CPU) {
    data_ = malloc(sizeBytes);
    if (!data_) {
      throw std::bad_alloc();
    }
  } else {
    // Handle other devices like GPU (e.g., cudaMalloc)
    throw std::runtime_error("Device not supported yet");
  }
}

Storage::Storage(void *data, size_t sizeBytes, hardware::Device device)
    : data_(data), sizeBytes_(sizeBytes), device_(device), ownsData_(false) {}

/**
 * @brief Destroys the Storage, freeing the allocated memory if it owns it.
 */
Storage::~Storage() {
  if (data_ && ownsData_) {
    if (device_.type == hardware::DeviceType::CPU) {
      free(data_);
    } else {
      // Handle other devices (e.g., cudaFree)
    }
  }
}

}  // namespace ir
}  // namespace mrt
