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
#include <vector>
#include <utility>
#ifdef ENABLE_TORCH_NPU
#include "torch_npu/csrc/aten/common/from_blob.h"
#include "torch_npu/csrc/core/npu/NPUFormat.h"
#endif
#include "hardware/hardware_abstract/device_context.h"
#include "hardware/hardware_abstract/collective/collective_manager.h"
#include "hardware/hardware_abstract/device_context_manager.h"
#include "ir/common/intrusive_ptr.h"
#include "ir/tensor/tensor.h"
#include "ops/utils/data_convert.h"
#include "common/logger.h"

namespace py = pybind11;
namespace ir = mrt::ir;

namespace mrt::ops {

static const std::map<at::ScalarType, ir::DataType> kAtScalarTypeToDataTypeMap = {
  {at::kHalf, ir::DataType::Type::Float16},
  {at::kBFloat16, ir::DataType::Type::BFloat16},
  {at::kFloat, ir::DataType::Type::Float32},
  {at::kDouble, ir::DataType::Type::Float64},
  {at::kComplexFloat, ir::DataType::Type::Complex64},
  {at::kChar, ir::DataType::Type::Int8},
  {at::kShort, ir::DataType::Type::Int16},
  {at::kInt, ir::DataType::Type::Int32},
  {at::kLong, ir::DataType::Type::Int64},
  {at::kByte, ir::DataType::Type::UInt8},
  {at::kBool, ir::DataType::Type::Bool},
};

static const std::map<ir::DataType, at::ScalarType> kDataTypeToAtScalarTypeMap = {
  {ir::DataType::Type::Float16, at::kHalf},
  {ir::DataType::Type::BFloat16, at::kBFloat16},
  {ir::DataType::Type::Float32, at::kFloat},
  {ir::DataType::Type::Float64, at::kDouble},
  {ir::DataType::Type::Complex64, at::kComplexFloat},
  {ir::DataType::Type::Int8, at::kChar},
  {ir::DataType::Type::Int16, at::kShort},
  {ir::DataType::Type::Int32, at::kInt},
  {ir::DataType::Type::Int64, at::kLong},
  {ir::DataType::Type::UInt8, at::kByte},
  {ir::DataType::Type::Bool, at::kBool},
};

ir::DataType FromTorchDType(const at::ScalarType &type) {
  auto iter = kAtScalarTypeToDataTypeMap.find(type);
  if (iter == kAtScalarTypeToDataTypeMap.end()) {
    LOG_EXCEPTION << "Unsupported at::ScalarType" << type << "for conversion to ir::DataType";
    return ir::DataType::Unknown;
  }

  return iter->second;
}

at::ScalarType ToTorchDType(ir::DataType type) {
  auto iter = kDataTypeToAtScalarTypeMap.find(type);
  if (iter == kDataTypeToAtScalarTypeMap.end()) {
    LOG_EXCEPTION << "Unsupported ir::DataType " << type << " for conversion to at::ScalarType";
    return at::kFloat;
  }

  return iter->second;
}

// Device conversion utilities
hardware::Device FromTorchDevice(const at::Device &device) {
  hardware::DeviceType deviceType = hardware::DeviceType::CPU;
  switch (device.type()) {
    case at::DeviceType::CPU:
      deviceType = hardware::DeviceType::CPU;
      break;
    case at::DeviceType::PrivateUse1:
      deviceType = hardware::DeviceType::NPU;
      break;
    default:
      LOG_EXCEPTION << "Unsupported torch::Device " << device.str() << " for conversion to hardware::Device";
  }
  return hardware::Device(deviceType, device.index());
}

at::Device ToTorchDevice(const hardware::Device device) {
  at::DeviceType deviceType = at::kCPU;
  switch (device.type) {
    case hardware::DeviceType::CPU:
      deviceType = at::kCPU;
      break;
#ifdef ENABLE_TORCH_NPU
    case hardware::DeviceType::NPU:
      deviceType = at::kPrivateUse1;
      break;
#endif
    default:
      LOG_EXCEPTION << "Unsupported hardware::DeviceType " << hardware::GetDeviceNameByType(device.type)
                    << " for conversion to torch::Device";
  }
  return at::Device(deviceType, device.index);
}

// Create a new mrt Tensor with a weak ref to torch Tensor data
ir::TensorPtr FromTorchTensor(const at::Tensor &tensor, bool isFake) {
  ir::DataType type = FromTorchDType(tensor.scalar_type());
  std::vector<int64_t> shape;
  shape.reserve(tensor.dim());
  for (auto &dim : tensor.sym_sizes()) {
    if (dim.is_symbolic()) {
      (void)shape.emplace_back(-1);
    } else if (dim.maybe_as_int().has_value()) {
      (void)shape.emplace_back(dim.maybe_as_int().value());
    } else {
      LOG_EXCEPTION << "Dynamic shape with non-int dimension is not supported";
    }
  }

  auto device = FromTorchDevice(tensor.device());
  if (isFake) {
    return ir::MakeIntrusive<ir::Tensor>(shape, type, device);
  } else {
    return ir::MakeIntrusive<ir::Tensor>(tensor.data_ptr(), shape, type, device);
  }
}

ir::StoragePtr CopyStorage(const ir::StoragePtr &srcStorage) {
  LOG_OUT << "Begin copy storage: " << srcStorage.get();
  auto device = srcStorage->GetDevice();
  auto storage = ir::MakeIntrusive<ir::Storage>(srcStorage->SizeBytes(), device);
  CHECK_IF_NULL(storage);
  storage->AllocateMemory();

  auto deviceId = mrt::collective::CollectiveManager::Instance().local_rank_id();
  mrt::device::DeviceContextKey deviceContextKey = {hardware::GetDeviceNameByType(device.type), deviceId};
  auto deviceContext = mrt::device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(deviceContextKey);
  CHECK_IF_NULL(deviceContext);
  CHECK_IF_NULL(deviceContext->deviceResManager_);
  if (!deviceContext->deviceResManager_->AsyncCopy(storage->Data(), srcStorage->Data(), srcStorage->SizeBytes(),
                                                   mrt::device::CopyType::D2D,
                                                   deviceContext->deviceResManager_->GetCurrentStream())) {
    LOG_EXCEPTION << "Async copy for output storage failed";
  }
  LOG_OUT << "End copy storage: " << srcStorage.get();
  return storage;
}

// Create a new torch Tensor by moving ownership of data from mrt Tensor
at::Tensor ToTorchTensor(const ir::TensorPtr &tensor) {
  CHECK_IF_NULL(tensor);
  auto storage = tensor->GetStorage();
  if (!storage->CheckOwnsData()) {
    // Parameter or tensor which references a parameter is graph output.
    storage = CopyStorage(storage);
  }
  auto allocator = storage->GetAllocator();
  auto deleter = [allocator](void *dataPtr) {
    if (dataPtr != nullptr) {
      allocator.Free(dataPtr);
    }
  };
  void *dataPtr = storage->Release();
  CHECK_IF_NULL(dataPtr);

  auto atDevice = ToTorchDevice(tensor->GetDevice());
  auto options = at::TensorOptions().dtype(ToTorchDType(tensor->Dtype())).device(atDevice);

  switch (atDevice.type()) {
    case at::DeviceType::CPU:
      return at::from_blob(dataPtr, tensor->Shape(), tensor->Strides(), std::move(deleter), options);
#ifdef ENABLE_TORCH_NPU
    case at::DeviceType::PrivateUse1:
      return at_npu::native::from_blob(dataPtr, tensor->Shape(), tensor->Strides(), tensor->StorageOffset(),
                                       std::move(deleter), options);
#endif
    default:
      LOG_EXCEPTION << "Unsupported DeviceType " << atDevice.str();
  }
  return at::Tensor{};
}

}  // namespace mrt::ops
