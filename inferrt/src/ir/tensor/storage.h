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

#ifndef __IR_TENSOR_STORAGE_H__
#define __IR_TENSOR_STORAGE_H__

#include <cstddef>
#include <memory>
#include <cstring>

#include "common/common.h"
#include "common/visible.h"
#include "hardware/device.h"
#include "ir/common/dtype.h"
#include "ir/common/intrusive_ptr.h"

namespace mrt {
namespace device {
class DeviceResManager;
}
class MRT_EXPORT Allocator {
 public:
  Allocator() = delete;
  Allocator(hardware::Device device);

  void *Allocate(size_t sizeBytes) const;
  void Free(void *ptr) const;

 private:
  device::DeviceResManager *deviceResManager_{nullptr};
};

namespace ir {

/**
 * @brief Implementation of the storage for a tensor.
 *
 * This class manages a block of memory on a specific device.
 * It is reference-counted and managed by the Storage class.
 */
class MRT_EXPORT Storage : public RefCounted {
 public:
  /**
   * @brief Constructs a Storage, allocating memory.
   * @param sizeBytes The size of the storage in bytes.
   * @param device The device where the storage is located.
   */
  Storage(size_t sizeBytes, hardware::Device device);
  /**
   * @brief Constructs a Storage from an existing buffer.
   * The storage does not own the data and will not free it.
   * @param data Pointer to the existing data.
   * @param sizeBytes The size of the storage in bytes.
   * @param device The device where the storage is located.
   */
  Storage(void *data, size_t sizeBytes, hardware::Device device);
  /**
   * @brief Destructor. Frees the allocated memory if it owns it.
   */
  ~Storage();

  /**
   * @brief Gets a const pointer to the data.
   * @return A const void pointer to the data.
   */
  const void *Data() const { return data_; }
  /**
   * @brief Gets a pointer to the data.
   * @return A void pointer to the data.
   */
  void *Data() { return data_; }
  /**
   * @brief Gets the size of the storage in bytes.
   * @return The size in bytes.
   */
  size_t SizeBytes() const { return sizeBytes_; }
  /**
   * @brief Gets the device of the storage.
   * @return The device.
   */
  hardware::Device GetDevice() const { return device_; }

  void SetData(void *data) {
    CHECK_IF_FAIL(!canOwnData_);
    data_ = data;
  }

  void Resize(size_t sizeBytes);

  /**
   * @brief Retrieves the allocator instance associated with this object.
   * @return The allocator used for memory management.
   */
  Allocator GetAllocator() const { return alloc_; }

  /**
   * @brief Allocates memory using the configured allocator according to device type.
   * This function checks for duplicate memory allocation or memory leaks.
   */
  void AllocateMemory();

  /**
   * @brief Frees the currently allocated memory, if owned.
   */
  void FreeMemory();

  /**
   * @brief Check whether this Storage can own data.
   * If true, the buffer pointed to by data_ is managed by this Storage object.
   */
  bool CheckCanOwnData() const { return canOwnData_; }

  /**
   * @brief Releases ownership of the managed pointer.
   * @return The raw data pointer.
   */
  void *Release();

 private:
  void *data_{nullptr};  ///< Pointer to the allocated memory.
  size_t sizeBytes_{0};  ///< Size of the memory in bytes.
  Allocator alloc_;
  hardware::Device device_;  ///< The device where the memory is allocated.
  const bool canOwnData_;    ///< Whether the storage can own data.
};

using StoragePtr = IntrusivePtr<Storage>;

}  // namespace ir
}  // namespace mrt

#endif  // __IR_TENSOR_STORAGE_H__