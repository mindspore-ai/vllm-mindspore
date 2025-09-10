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

#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "ir/common/intrusive_ptr.h"
#include "ir/tensor/tensor.h"
#include "common/logger.h"

namespace py = pybind11;
using namespace mrt;

namespace {
// DataType conversion utilities
ir::DataType FromTorchDType(const at::ScalarType &type) {
  switch (type) {
    case at::kFloat:
      return ir::DataType::Float32;
    case at::kDouble:
      return ir::DataType::Float64;
    case at::kInt:
      return ir::DataType::Int32;
    case at::kLong:
      return ir::DataType::Int64;
    case at::kShort:
      return ir::DataType::Int16;
    case at::kBool:
      return ir::DataType::Bool;
    default:
      LOG_EXCEPTION << "Unsupported at::ScalarType for conversion to ir::DataType";
      return ir::DataType::Unknown;
  }
}

at::ScalarType ToTorchDType(ir::DataType type) {
  switch (type) {
    case ir::DataType::Float32:
      return at::kFloat;
    case ir::DataType::Float64:
      return at::kDouble;
    case ir::DataType::Int32:
      return at::kInt;
    case ir::DataType::Int64:
      return at::kLong;
    case ir::DataType::Int16:
      return at::kShort;
    case ir::DataType::Bool:
      return at::kBool;
    default:
      LOG_EXCEPTION << "Unsupported ir::DataType for conversion to at::ScalarType";
      return at::kFloat;
  }
}

// Device conversion utilities
hardware::Device FromTorchDevice(const at::Device &device) {
  hardware::DeviceType deviceType;
  switch (device.type()) {
    case at::DeviceType::CPU:
      deviceType = hardware::DeviceType::CPU;
      break;
    default:
      LOG_EXCEPTION << "Unsupported torch::Device for conversion to hardware::Device";
  }
  return hardware::Device(deviceType, device.index());
}

at::Device ToTorchDevice(const hardware::Device device) {
  at::DeviceType deviceType;
  switch (device.type) {
    case hardware::DeviceType::CPU:
      deviceType = at::kCPU;
      break;
    default:
      LOG_EXCEPTION << "Unsupported hardware::DeviceType for conversion to torch::Device";
  }
  return at::Device(deviceType, device.index);
}

// Create a new mrt Tensor without owns data
ir::TensorPtr FromTorchTensor(const at::Tensor &tensor, bool isFake = false) {
  ir::DataType type = FromTorchDType(tensor.scalar_type());
  std::vector<int64_t> shape(tensor.sizes().begin(), tensor.sizes().end());
  auto device = FromTorchDevice(tensor.device());
  void *data = isFake ? nullptr : tensor.data_ptr();
  return ir::MakeIntrusive<ir::Tensor>(data, shape, type, device);
}

// Create a new torch Tensor with shared data
at::Tensor ToTorchTensor(const ir::TensorPtr &tensor) {
  CHECK_IF_NULL(tensor);
  auto options = at::TensorOptions().dtype(ToTorchDType(tensor->Dtype())).device(ToTorchDevice(tensor->GetDevice()));
  const auto &storage = tensor->GetStorage();
  // Only support operator output as graph output case currently.
  CHECK_IF_FAIL(storage->CheckOwnsData());
  auto allocator = storage->GetAllocator();
  void *dataPtr = storage->Release();
  CHECK_IF_NULL(dataPtr);
  return at::from_blob(
    const_cast<void *>(dataPtr), tensor->Shape(), tensor->Strides(),
    [allocator](void *dataPtr) { allocator.Free(dataPtr); }, options);
}

void UpdateTensorData(ir::Tensor &self, const at::Tensor &atTensor) {
  ir::DataType type = FromTorchDType(atTensor.scalar_type());
  std::vector<int64_t> shape(atTensor.sizes().begin(), atTensor.sizes().end());
  void *data = atTensor.data_ptr();

  if (self.GetDevice() != FromTorchDevice(atTensor.device())) {
    LOG_EXCEPTION << "Device mismatch in update_tensor_data";
  }

  self.SetDtype(type);
  self.SetShape(std::move(shape));
  self.ResizeStorage();
  self.UpdateData(data);
}
}  // namespace

PYBIND11_DECLARE_HOLDER_TYPE(T, ir::IntrusivePtr<T>, true);

PYBIND11_MODULE(_mrt_torch, m) {
  m.doc() = "PyTorch extension for MRT";
  m.def("from_torch", &FromTorchTensor, py::arg("tensor"), py::arg("is_fake") = false);
  m.def("to_torch", &ToTorchTensor, py::return_value_policy::reference);
  m.def("update_tensor_data", &UpdateTensorData);
}
