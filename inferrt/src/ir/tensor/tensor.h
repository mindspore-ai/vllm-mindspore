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

#ifndef __IR_TENSOR_TENSOR_H__
#define __IR_TENSOR_TENSOR_H__

#include <vector>
#include <numeric>

#include "hardware/device.h"
#include "ir/common/dtype.h"
#include "ir/tensor/storage.h"

namespace mrt {
namespace ir {

/**
 * @brief A multi-dimensional array (tensor).
 *
 * This class holds the metadata of a tensor, such as its dimensions, data type,
 * and a reference to the underlying storage.
 */
class Tensor {
 public:
  /**
   * @brief Constructs an empty Tensor with uninitialized data.
   * A new storage is allocated for the tensor.
   * @param shape The dimensions of the tensor.
   * @param dtype The data type of the tensor.
   * @param device The device to allocate the tensor on.
   * @return The newly created tensor.
   */
  Tensor(const std::vector<int64_t> &shape, DataType dtype, hardware::Device device);
  /**
   * @brief Constructs a Tensor from an existing Storage.
   * @param storage The underlying storage for the tensor data.
   * @param dtype The data type of the tensor elements.
   * @param shape The dimensions of the tensor.
   */
  Tensor(StoragePtr storage, DataType dtype, const std::vector<int64_t> &shape);
  /**
   * @brief Constructs a Tensor from an existing data blob.
   * The tensor does not own the memory.
   * @param data Pointer to the data.
   * @param shape The dimensions of the tensor.
   * @param dtype The data type of the tensor.
   * @param device The device where the data is located.
   * @return The newly created tensor.
   */
  Tensor(void *data, const std::vector<int64_t> &shape, DataType dtype, hardware::Device device);

  /**
   * @brief Move constructor.
   */
  Tensor(Tensor &&other) noexcept {
    dtype_ = other.dtype_;
    numel_ = other.numel_;
    storageOffset_ = other.storageOffset_;
    shape_ = std::move(other.shape_);
    strides_ = std::move(other.strides_);
    storage_ = std::move(other.storage_);

    // Invalidate the moved-from tensor
    other.dtype_ = DataType::Unknown;
    other.numel_ = 0;
    other.storageOffset_ = 0;
  }

  /**
   * @brief Deleted copy constructor.
   */
  Tensor(const Tensor &) = delete;
  Tensor &operator=(const Tensor &) = delete;

  /**
   * @brief Gets the data type of the tensor.
   * @return The data type.
   */
  DataType Dtype() const { return dtype_; }
  /**
   * @brief Gets the dimensions of the tensor.
   * @return A const reference to the vector of dimensions.
   */
  const std::vector<int64_t> &Shape() const { return shape_; }
  /**
   * @brief Gets the strides of the tensor.
   * @return A const reference to the vector of strides.
   */
  const std::vector<int64_t> &Strides() const { return strides_; }
  /**
   * @brief Gets the number of dimensions of the tensor.
   * @return The number of dimensions.
   */
  int64_t Dim() const { return shape_.size(); }
  /**
   * @brief Gets the total number of elements in the tensor.
   * @return The number of elements, or -1 for dynamic shapes.
   */
  int64_t Numel() const { return numel_; }
  /**
   * @brief Checks if the tensor has a dynamic shape.
   * @return true if the shape is dynamic, false otherwise.
   */
  bool HasDynamicShape() const { return numel_ < 0; }
  /**
   * @brief Gets the device where the tensor data is stored.
   * @return The device.
   */
  hardware::Device GetDevice() const { return storage_->GetDevice(); }
  /**
   * @brief Gets the underlying storage of the tensor.
   * @return The storage.
   */
  StoragePtr GetStorage() const { return storage_; }
  /**
   * @brief Gets a raw const pointer to the tensor's data.
   * This pointer takes into account the storage offset.
   * @return A const void pointer to the data.
   */
  const void *DataPtr() const {
    return static_cast<const char *>(storage_->Data()) + storageOffset_ * dtype_.GetSize();
  }
  /**
   * @brief Gets a raw pointer to the tensor's data.
   * This pointer takes into account the storage offset.
   * @return A void pointer to the data.
   */
  void *DataPtr() { return static_cast<char *>(storage_->Data()) + storageOffset_ * dtype_.GetSize(); }
  /**
   * @brief Sets the data type of the tensor.
   * @param dtype The new data type to set.
   */
  void SetDtype(DataType dtype) { dtype_ = dtype; }
  /**
   * @brief Sets the shape of the tensor.
   * @param shape The new shape to set.
   */
  void SetShape(const std::vector<int64_t> &shape);
  /**
   * @brief Sets the shape of the tensor.
   * @param shape The new shape to set.
   */
  void SetShape(const std::vector<int64_t> &&shape);

 private:
  /**
   * @brief Computes the strides from the dimensions.
   */
  void ComputeStrides();

  DataType dtype_;                ///< The data type of the elements.
  std::vector<int64_t> shape_;    ///< The dimensions of the tensor.
  std::vector<int64_t> strides_;  ///< The strides of the tensor.
  int64_t numel_ = 0;             ///< The total number of elements.
  StoragePtr storage_{nullptr};   ///< The underlying storage.
  int64_t storageOffset_ = 0;     ///< The offset in the storage, in number of elements.
};

}  // namespace ir
}  // namespace mrt

#endif  // __IR_TENSOR_TENSOR_H__
