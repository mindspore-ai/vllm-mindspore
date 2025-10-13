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
#include "hardware/hardware_abstract/device_context_manager.h"

namespace mrt {
Allocator::Allocator(hardware::Device device) {
  device::DeviceContextKey deviceContextKey = device::DeviceToDeviceContextKey(device);
  auto deviceContext = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(deviceContextKey);
  deviceResManager_ = deviceContext->deviceResManager_.get();
}

void *Allocator::Allocate(size_t sizeBytes) const { return deviceResManager_->AllocateMemory(sizeBytes); }

void Allocator::Free(void *ptr) const { deviceResManager_->FreeMemory(ptr); }

namespace ir {
Storage::Storage(size_t sizeBytes, hardware::Device device)
    : sizeBytes_(sizeBytes), alloc_(device), device_(device), canOwnData_(true) {
  Resize(sizeBytes_);
}

Storage::Storage(void *data, size_t sizeBytes, hardware::Device device)
    : data_(data), sizeBytes_(sizeBytes), alloc_(device), device_(device), canOwnData_(false) {}

Storage::~Storage() {
  if (canOwnData_ && data_ != nullptr) {
    alloc_.Free(data_);
  }
}

void Storage::Resize(size_t sizeBytes) {
  sizeBytes_ = sizeBytes;
  if (!canOwnData_) {
    return;
  }
  if (data_ != nullptr) {
    LOG_EXCEPTION << "Device memory leak detected, device type: " << GetDeviceNameByType(device_.type);
  }
}

void Storage::AllocateMemory() {
  if (!canOwnData_) {
    LOG_EXCEPTION << "Can not allocate memory for a storage which can not own data, this Storage is used to "
                     "reference memory passed in from external sources. The device type: "
                  << GetDeviceNameByType(device_.type) << ", data: " << data_;
  }

  if (data_ != nullptr) {
    LOG_EXCEPTION << "Device memory has already been allocated, or a device memory leak has occurred, device type: "
                  << GetDeviceNameByType(device_.type) << ", data: " << data_;
  }
  data_ = alloc_.Allocate(sizeBytes_);
  CHECK_IF_NULL(data_);
}

void Storage::FreeMemory() {
  if (!canOwnData_) {
    LOG_EXCEPTION << "Can not free memory for a storage which can not own data, this Storage is used to "
                     "reference memory passed in from external sources.";
  }

  CHECK_IF_NULL(data_);
  alloc_.Free(data_);
  data_ = nullptr;
}

void *Storage::Release() {
  if (!canOwnData_) {
    LOG_EXCEPTION << "Can not release memory to other for a storage which can not own data, this Storage is used to "
                     "reference memory passed in from external sources.";
  }
  void *p = data_;
  data_ = nullptr;
  return p;
}
}  // namespace ir
}  // namespace mrt
