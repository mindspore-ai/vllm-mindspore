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
#include "ir/common/intrusive_ptr.h"
#include "ir/common/dtype.h"
#include "ir/tensor/storage.h"

namespace mrt {
namespace ir {

/**
 * @brief Implementation of a multi-dimensional array (tensor).
 *
 * This class holds the metadata of a tensor, such as its dimensions, data type,
 * and a reference to the underlying storage. It is reference-counted and managed
 * by the Tensor class.
 */
class TensorImpl : public RefCounted {
 public:
  /**
   * @brief Constructs a TensorImpl.
   * @param storage The underlying storage for the tensor data.
   * @param dtype The data type of the tensor elements.
   * @param shape The dimensions of the tensor.
   */
  TensorImpl(Storage storage, DataType dtype, const std::vector<int64_t> &shape);

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
  DataType Dtype() const { return dtype_; }
  /**
   * @brief Gets the device where the tensor data is stored.
   * @return The device.
   */
  bool HasDynamicShape() const { return numel_ < 0; }
  /**
   * @brief Gets the data type of the tensor.
   * @return The data type.
   */
  hardware::Device GetDevice() const { return storage_.GetDevice(); }
  /**
   * @brief Gets the underlying storage of the tensor.
   * @return The storage.
   */
  Storage GetStorage() const { return storage_; }
  /**
   * @brief Gets a raw pointer to the tensor's data.
   * This pointer takes into account the storage offset.
   * @return A void pointer to the data.
   */
  void *DataPtr() const { return static_cast<char *>(storage_.Data()) + storageOffset_ * dtype_.GetSize(); }

 private:
  /**
   * @brief Computes the strides from the dimensions.
   */
  void ComputeStrides();

  Storage storage_;  ///< The underlying storage.
  DataType dtype_;   ///< The data type of the elements.
  std::vector<int64_t> shape_;
  std::vector<int64_t> strides_;
  int64_t storageOffset_ = 0;  ///< The offset in the storage, in number of elements.
  int64_t numel_ = 0;          ///< The total number of elements.
};

/**
 * @brief A handle to a TensorImpl.
 *
 * This class provides a user-friendly interface to the tensor implementation.
 * It uses an IntrusivePtr to manage the lifetime of the TensorImpl.
 */
class Tensor {
 public:
  /**
   * @brief Default constructor. Creates an uninitialized Tensor.
   */
  Tensor() = default;

  /**
   * @brief Constructs a Tensor.
   * @param storage The underlying storage for the tensor data.
   * @param dtype The data type of the tensor elements.
   * @param shape The dimensions of the tensor.
   */
  Tensor(Storage storage, DataType dtype, const std::vector<int64_t> &shape)
      : impl_(MakeIntrusive<TensorImpl>(std::move(storage), dtype, shape)) {}

  /**
   * @brief Gets the dimensions of the tensor.
   * @return A const reference to the vector of dimensions.
   */
  const std::vector<int64_t> &Shape() const { return impl_->Shape(); }
  /**
   * @brief Gets the strides of the tensor.
   * @return A const reference to the vector of strides.
   */
  const std::vector<int64_t> &Strides() const { return impl_->Strides(); }
  /**
   * @brief Gets the number of dimensions of the tensor.
   * @return The number of dimensions.
   */
  int64_t Dim() const { return impl_->Dim(); }
  /**
   * @brief Gets the total number of elements in the tensor.
   * @return The number of elements, or -1 for dynamic shapes.
   */
  int64_t Numel() const { return impl_->Numel(); }
  /**
   * @brief Checks if the tensor has a dynamic shape.
   * @return true if the shape is dynamic, false otherwise.
   */
  bool HasDynamicShape() const { return impl_->HasDynamicShape(); }
  /**
   * @brief Gets the data type of the tensor.
   * @return The data type.
   */
  DataType Dtype() const { return impl_->Dtype(); }
  /**
   * @brief Gets the device where the tensor data is stored.
   * @return The device.
   */
  hardware::Device GetDevice() const { return impl_->GetDevice(); }
  /**
   * @brief Gets a raw pointer to the tensor's data.
   * @return A void pointer to the data.
   */
  void *DataPtr() const { return impl_->DataPtr(); }

  /**
   * @brief Gets a typed pointer to the tensor's data.
   * @tparam T The desired data type.
   * @return A typed pointer to the data.
   */
  template <typename T>
  T *Data() const {
    return static_cast<T *>(DataPtr());
  }

  /**
   * @brief Checks if the tensor is defined (not null).
   * @return true if the tensor is defined, false otherwise.
   */
  bool Defined() const { return bool(impl_); }

  /**
   * @brief Gets the underlying storage of the tensor.
   * @return The storage.
   */
  Storage GetStorage() const { return impl_->GetStorage(); }

 private:
  IntrusivePtr<TensorImpl> impl_;
};

/**
 * @brief Creates an empty tensor with uninitialized data.
 * @param shape The dimensions of the tensor.
 * @param dtype The data type of the tensor.
 * @param device The device to allocate the tensor on.
 * @return The newly created tensor.
 */
Tensor Empty(const std::vector<int64_t> &shape, DataType dtype, hardware::Device device);

/**
 * @brief Creates a tensor from an existing data blob.
 * The tensor does not own the memory.
 * @param data Pointer to the data.
 * @param shape The dimensions of the tensor.
 * @param dtype The data type of the tensor.
 * @param device The device where the data is located.
 * @return The newly created tensor.
 */
Tensor FromBlob(void *data, const std::vector<int64_t> &shape, DataType dtype, hardware::Device device);

std::ostream &operator<<(std::ostream &os, const Tensor &tensor);

}  // namespace ir
}  // namespace mrt

#endif  // __IR_TENSOR_TENSOR_H__
