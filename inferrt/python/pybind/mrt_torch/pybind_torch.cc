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

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <pybind11/pybind11.h>  // Bridge
#include <torch/extension.h>
#include <vector>
#include <utility>

#ifdef ENABLE_TORCH_NPU
#include "acl/acl.h"
#include "torch_npu/csrc/aten/common/from_blob.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/NPUFormat.h"
#include "torch_npu/csrc/core/NPUStorageImpl.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#endif

#include "common/intrusive_ptr_caster.h"
#include "common/logger.h"
#include "ir/common/intrusive_ptr.h"
#include "ir/tensor/tensor.h"
#include "ir/value/value.h"
#include "ir/graph.h"
#include "ops/utils/async.h"
#include "hardware/hardware_abstract/device_context.h"
#include "hardware/hardware_abstract/collective/collective_manager.h"
#include "hardware/hardware_abstract/device_context_manager.h"

namespace nb = nanobind;
namespace ir = mrt::ir;
namespace hardware = mrt::hardware;

PYBIND11_DECLARE_HOLDER_TYPE(T, ir::IntrusivePtr<T>, true);

namespace {
// DataType conversion utilities

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

#ifdef ENABLE_TORCH_NPU
inline aclFormat ConvertMemoryFormatToAclFormat(ir::MemoryFormat format) {
  static const std::map<ir::MemoryFormat, aclFormat> kMemoryFormatToAclFormatMap = {
    {ir::MemoryFormat::FORMAT_UNDEFINED, ACL_FORMAT_UNDEFINED},
    {ir::MemoryFormat::FORMAT_NCHW, ACL_FORMAT_NCHW},
    {ir::MemoryFormat::FORMAT_NHWC, ACL_FORMAT_NHWC},
    {ir::MemoryFormat::FORMAT_ND, ACL_FORMAT_ND},
    {ir::MemoryFormat::FORMAT_NC1HWC0, ACL_FORMAT_NC1HWC0},
    {ir::MemoryFormat::FORMAT_FRACTAL_Z, ACL_FORMAT_FRACTAL_Z},
    {ir::MemoryFormat::FORMAT_NC1HWC0_C04, ACL_FORMAT_NC1HWC0_C04},
    {ir::MemoryFormat::FORMAT_HWCN, ACL_FORMAT_HWCN},
    {ir::MemoryFormat::FORMAT_NDHWC, ACL_FORMAT_NDHWC},
    {ir::MemoryFormat::FORMAT_FRACTAL_NZ, ACL_FORMAT_FRACTAL_NZ},
    {ir::MemoryFormat::FORMAT_NCDHW, ACL_FORMAT_NCDHW},
    {ir::MemoryFormat::FORMAT_NDC1HWC0, ACL_FORMAT_NDC1HWC0},
    {ir::MemoryFormat::FORMAT_FRACTAL_Z_3D, ACL_FRACTAL_Z_3D},
    {ir::MemoryFormat::FORMAT_NC, ACL_FORMAT_NC},
    {ir::MemoryFormat::FORMAT_NCL, ACL_FORMAT_NCL},
  };

  auto iter = kMemoryFormatToAclFormatMap.find(format);
  if (iter == kMemoryFormatToAclFormatMap.end()) {
    LOG_EXCEPTION << "Unsupported MemoryFormat " << format << " for conversion to aclFormat";
    return ACL_FORMAT_UNDEFINED;
  }

  return iter->second;
}
#endif

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
  hardware::DeviceType deviceType;
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
  at::DeviceType deviceType;
  switch (device.type) {
    case hardware::DeviceType::CPU:
      deviceType = at::kCPU;
      break;
    case hardware::DeviceType::NPU:
      deviceType = at::kPrivateUse1;
      break;
    default:
      LOG_EXCEPTION << "Unsupported hardware::DeviceType " << hardware::GetDeviceNameByType(device.type)
                    << " for conversion to torch::Device";
  }
  return at::Device(deviceType, device.index);
}

// Create a new mrt Tensor with a weak ref to torch Tensor data
ir::TensorPtr FromTorchTensor(const at::Tensor &tensor, bool isFake = false) {
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
    auto &waitLaunchFinish = mrt::ops::OpAsync::GetWaitLaunchFinishFunc();
    if (waitLaunchFinish != nullptr) {
      waitLaunchFinish();
    }
    // Parameter or tensor which references a parameter is graph output.
    storage = CopyStorage(storage);
  }

  void *dataPtr = storage->Data();
  CHECK_IF_NULL(dataPtr);
  auto deleterFn = storage->GetDeleter();
  std::function<void(void *)> deleter;
  if (deleterFn == nullptr) {
    auto allocator = storage->GetAllocator();
    deleter = [allocator, dataPtr](void *) {
      if (dataPtr != nullptr) {
        allocator.Free(dataPtr);
      }
    };
  } else {
    deleter = [deleterFn, dataPtr](void *) {
      if (dataPtr != nullptr) {
        deleterFn(dataPtr);
      }
    };
  }
  storage->Release();

  auto atDevice = ToTorchDevice(tensor->GetDevice());
  auto options = at::TensorOptions().dtype(ToTorchDType(tensor->Dtype())).device(atDevice);
  if (tensor->Numel() == 0) {
    return at::empty(tensor->Shape(), options);
  }

  switch (atDevice.type()) {
    case at::DeviceType::CPU: {
      if (tensor->Strides().empty()) {
        return at::from_blob(dataPtr, tensor->Shape(), std::move(deleter), options);
      }
      return at::from_blob(dataPtr, tensor->Shape(), tensor->Strides(), std::move(deleter), options);
    }
#ifdef ENABLE_TORCH_NPU
    case at::DeviceType::PrivateUse1: {
      at::Tensor out;
      if (tensor->Strides().empty()) {
        out = at_npu::native::from_blob(
          static_cast<char *>(dataPtr) + tensor->StorageOffset() * (tensor->Dtype().GetSize()), tensor->Shape(),
          std::move(deleter), options);
      } else {
        out = at_npu::native::from_blob(
          static_cast<char *>(dataPtr) + tensor->StorageOffset() * (tensor->Dtype().GetSize()), tensor->Shape(),
          tensor->Strides(), 0, std::move(deleter), options);
      }
      auto &desc = static_cast<torch_npu::NPUStorageImpl *>(out.storage().unsafeGetStorageImpl())->npu_desc_;
      desc.npu_format_ = ConvertMemoryFormatToAclFormat(tensor->Format());
      return out;
    }
#endif
    default:
      LOG_EXCEPTION << "Unsupported DeviceType " << atDevice.str();
  }
}

void UpdateTensor(const ir::TensorPtr &self, nb::handle h) {
  self->SetUpdater([h](ir::Tensor *tensor) {
    pybind11::handle ph(h.ptr());
    at::Tensor atTensor = ph.cast<at::Tensor>();

    ir::DataType type = FromTorchDType(atTensor.scalar_type());
    std::vector<int64_t> shape(atTensor.sizes().begin(), atTensor.sizes().end());
    void *data = atTensor.data_ptr();

    auto device = tensor->GetDevice();
    if (device != FromTorchDevice(atTensor.device())) {
      LOG_EXCEPTION << "Device mismatch in update_tensor";
    }

#ifdef ENABLE_TORCH_NPU
    if (device.type == hardware::DeviceType::NPU) {
      auto npuFormat = at_npu::native::get_npu_format(atTensor);
      tensor->SetFormat(static_cast<ir::MemoryFormat>(npuFormat));
      tensor->SetStrides(atTensor.strides().vec());
      tensor->SetStorageOffset(atTensor.storage_offset());
      tensor->SetStorageShape(at_npu::native::get_npu_storage_sizes(atTensor));
      LOG_OUT << "Update tensor, format=" << ir::FormatEnumToStr(tensor->Format()) << ", strides=" << tensor->Strides()
              << ", storageOffset=" << tensor->StorageOffset() << ", storageShape=" << tensor->StorageShape()
              << ", isView=" << atTensor.is_view() << " at.tensor.shape: " << atTensor.sizes();
    }
#endif

    tensor->SetOwnsStorage(false);
    tensor->SetDtype(type);
    tensor->SetShape(std::move(shape));
    tensor->Resize();
    CHECK_IF_NULL(data);
    tensor->UpdateData(data);
    tensor->GetStorage()->Resize(atTensor.storage().nbytes());
  });
}

void UpdateMrtValue(const ir::ValuePtr &mrtValue, nb::handle h) {
  CHECK_IF_NULL(mrtValue);

  switch (mrtValue->GetTag()) {
    case ir::Value::Tag::Tensor: {
      UpdateTensor(mrtValue->ToTensor(), h);
      return;
    }
    case ir::Value::Tag::Symbol: {
      const int64_t value = nb::cast<int64_t>(h);
      auto symbolicExpr = mrtValue->ToSymbol();
      if (symbolicExpr->GetKind() == ir::SymbolicExpr::Kind::Variable) {
        auto symbolicVar = static_cast<ir::SymbolicVar *>(symbolicExpr.get());
        symbolicVar->SetValue(value);
      }
      return;
    }
    case ir::Value::Tag::Tuple: {
      auto mrtTuple = mrtValue->ToTuple();
      auto pyTuple = nb::cast<nb::tuple>(h);
      if (mrtTuple->Size() != pyTuple.size()) {
        LOG_EXCEPTION << "Expected " << mrtTuple->Size() << " items in tuple, but received " << pyTuple.size();
      }
      auto it = mrtTuple->begin();
      for (size_t i = 0; i < mrtTuple->Size(); ++i, ++it) {
        UpdateMrtValue(*it, pyTuple[i]);
      }
      return;
    }
    case ir::Value::Tag::Int: {
      *mrtValue = ir::Value(nb::cast<int64_t>(h));
      return;
    }
    case ir::Value::Tag::Double: {
      *mrtValue = ir::Value(nb::cast<double>(h));
      return;
    }
    case ir::Value::Tag::Bool: {
      *mrtValue = ir::Value(nb::cast<bool>(h));
      return;
    }
    case ir::Value::Tag::String: {
      *mrtValue = ir::Value(nb::cast<std::string>(h));
      return;
    }
    case ir::Value::Tag::None: {
      return;
    }
    default:
      LOG_EXCEPTION << "Unsupported Value Tag";
      return;
  }
}

void BatchUpdateRuntimeInputs(const nb::list &paramNodes, const nb::tuple &newInputs) {
  if (paramNodes.size() != newInputs.size()) {
    LOG_EXCEPTION << "Expected " << paramNodes.size() << " inputs, but received " << newInputs.size();
  }

  for (size_t i = 0; i < paramNodes.size(); ++i) {
    const auto &mrtNode = nb::cast<ir::NodePtr>(paramNodes[i]);
    UpdateMrtValue(mrtNode->output, newInputs[i]);
  }
}

void SetDeviceContext() {
#ifdef ENABLE_TORCH_NPU
  mrt::device::DeviceContextKey deviceContextKey{"Ascend",
                                                 mrt::collective::CollectiveManager::Instance().local_rank_id()};
  auto deviceContext = mrt::device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(deviceContextKey);
  CHECK_IF_NULL(deviceContext);
  CHECK_IF_NULL(deviceContext->deviceResManager_);

  auto currentNPUStream = c10_npu::getCurrentNPUStream();
  auto bindStreamFunc = [currentNPUStream]() { c10_npu::setCurrentNPUStream(currentNPUStream); };
  deviceContext->deviceResManager_->SetBindStreamFunc(bindStreamFunc);

  if (mrt::ops::IsEnablePipeline()) {
    mrt::ops::OpAsync::SetLaunchOpFunc(at_npu::native::OpCommand::RunOpApiV2);
    mrt::ops::OpAsync::SetWaitLaunchFinishFunc([]() { (void)c10_npu::getCurrentNPUStream().stream(true); });
  }

  auto currentStream = currentNPUStream.stream(false);
  CHECK_IF_NULL(currentStream);
  deviceContext->deviceResManager_->SetCurrentStream(currentStream);
  auto ascend_allocator = [](size_t size) -> void * {
    void *cur_alloc = c10_npu::NPUCachingAllocator::raw_alloc(size);
    LOG_OUT << "Memory allocated via PyTorch, new addr: " << cur_alloc;
    return cur_alloc;
  };
  deviceContext->deviceResManager_->SetAllocator(ascend_allocator);
  auto ascend_deleter = [](void *dataPtr) {
    if (dataPtr != nullptr) {
      c10_npu::NPUCachingAllocator::raw_delete(dataPtr);
    }
  };
  deviceContext->deviceResManager_->SetDeleter(ascend_deleter);
#endif
}

// Wrappers for nanobind
ir::TensorPtr FromTorchTensorWrapper(nb::handle h, bool isFake) {
  pybind11::handle ph(h.ptr());
  at::Tensor t = ph.cast<at::Tensor>();
  return FromTorchTensor(t, isFake);
}

nb::object ToTorchTensorWrapper(const ir::TensorPtr &tensor) {
  at::Tensor t = ToTorchTensor(tensor);
  pybind11::object po = pybind11::cast(t);
  return nb::steal(po.release().ptr());
}
}  // namespace

NB_MODULE(_mrt_torch, m) {
  m.doc() = "PyTorch extension for MRT";
  m.def("from_torch", &FromTorchTensorWrapper, nb::arg("tensor"), nb::arg("is_fake") = false);
  m.def("to_torch", &ToTorchTensorWrapper, nb::rv_policy::reference);
  m.def("set_device_context", &SetDeviceContext);
  m.def("batch_update_runtime_inputs", &BatchUpdateRuntimeInputs, "Batch update runtime inputs for nodes");
}
