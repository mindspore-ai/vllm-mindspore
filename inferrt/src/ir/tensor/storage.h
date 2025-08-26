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

#include "hardware/device.h"
#include "ir/common/intrusive_ptr.h"
#include "ir/common/dtype.h"

namespace mrt {
namespace ir {

/**
 * @brief Implementation of the storage for a tensor.
 *
 * This class manages a block of memory on a specific device.
 * It is reference-counted and managed by the Storage class.
 */
class StorageImpl : public RefCounted {
 public:
  /**
   * @brief Constructs a StorageImpl, allocating memory.
   * @param sizeBytes The size of the storage in bytes.
   * @param device The device where the storage is located.
   */
  StorageImpl(size_t sizeBytes, hardware::Device device);
  /**
   * @brief Constructs a StorageImpl from an existing buffer.
   * The storage does not own the data and will not free it.
   * @param data Pointer to the existing data.
   * @param sizeBytes The size of the storage in bytes.
   * @param device The device where the storage is located.
   */
  StorageImpl(void *data, size_t sizeBytes, hardware::Device device);
  /**
   * @brief Destructor. Frees the allocated memory if it owns it.
   */
  ~StorageImpl();

  /**
   * @brief Gets a pointer to the data.
   * @return A void pointer to the data.
   */
  void *Data() const { return data_; }
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

 private:
  void *data_;               ///< Pointer to the allocated memory.
  size_t sizeBytes_;         ///< Size of the memory in bytes.
  hardware::Device device_;  ///< The device where the memory is allocated.
  bool ownsData_{true};      ///< Whether the storage owns the data.
};

/**
 * @brief A handle to a StorageImpl.
 *
 * This class provides a user-friendly interface to the storage implementation.
 * It uses an IntrusivePtr to manage the lifetime of the StorageImpl.
 */
class Storage {
 public:
  /**
   * @brief Default constructor. Creates an uninitialized Storage.
   */
  Storage() = default;

  /**
   * @brief Constructs a Storage, allocating memory.
   * @param sizeBytes The size of the storage in bytes.
   * @param device The device where the storage is located.
   */
  Storage(size_t sizeBytes, hardware::Device device) : impl_(MakeIntrusive<StorageImpl>(sizeBytes, device)) {}

  /**
   * @brief Constructs a Storage from an existing buffer.
   * The storage does not own the data and will not free it.
   * @param data Pointer to the existing data.
   * @param sizeBytes The size of the storage in bytes.
   * @param device The device where the storage is located.
   */
  Storage(void *data, size_t sizeBytes, hardware::Device device)
      : impl_(MakeIntrusive<StorageImpl>(data, sizeBytes, device)) {}

  /**
   * @brief Gets a pointer to the data.
   * @return A void pointer to the data.
   */
  void *Data() const { return impl_->Data(); }
  /**
   * @brief Gets the size of the storage in bytes.
   * @return The size in bytes.
   */
  size_t SizeBytes() const { return impl_->SizeBytes(); }
  /**
   * @brief Gets the device of the storage.
   * @return The device.
   */
  hardware::Device GetDevice() const { return impl_->GetDevice(); }
  /**
   * @brief Gets the underlying implementation pointer.
   * @return A const reference to the IntrusivePtr of the StorageImpl.
   */
  const IntrusivePtr<StorageImpl> &GetImpl() const { return impl_; }

 private:
  IntrusivePtr<StorageImpl> impl_;
};

}  // namespace ir
}  // namespace mrt

#endif  // __IR_TENSOR_STORAGE_H__
