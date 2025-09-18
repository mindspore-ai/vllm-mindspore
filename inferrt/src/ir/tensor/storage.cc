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

Storage::Storage(size_t sizeBytes, hardware::Device device)
    : sizeBytes_(sizeBytes), alloc_(device), device_(device), ownsData_(true) {
  Resize(sizeBytes_);
}

Storage::Storage(void *data, size_t sizeBytes, hardware::Device device)
    : data_(data), sizeBytes_(sizeBytes), alloc_(device), device_(device), ownsData_(data == nullptr) {}

Storage::~Storage() {
  if (ownsData_ && data_ != nullptr) {
    alloc_.Free(data_);
  }
}

void Storage::Resize(size_t sizeBytes) {
  sizeBytes_ = sizeBytes;
  if (!ownsData_) {
    return;
  }
  if (data_ != nullptr) {
    LOG_EXCEPTION << "Device memory leak detected, device type: " << GetDeviceNameByType(device_.type);
  }
}

void Storage::AllocateMemory() {
  if (data_ != nullptr && ownsData_) {
    LOG_EXCEPTION << "Device memory has already been allocated, or a device memory leak has occurred, device type: "
                  << GetDeviceNameByType(device_.type) << ", data: " << data_;
  }

  data_ = alloc_.Allocate(sizeBytes_);
  CHECK_IF_NULL(data_);
  ownsData_ = true;
}

void Storage::FreeMemory() {
  if (!ownsData_) {
    LOG_EXCEPTION << "Can not free memory for a storage which doesn't own data, this Storage may be used to "
                     "reference memory passed in from external sources.";
  }

  CHECK_IF_NULL(data_);
  alloc_.Free(data_);
  data_ = nullptr;
  ownsData_ = false;
}

void *Storage::Release() {
  void *p = data_;
  data_ = nullptr;
  ownsData_ = false;
  return p;
}
}  // namespace ir
}  // namespace mrt
