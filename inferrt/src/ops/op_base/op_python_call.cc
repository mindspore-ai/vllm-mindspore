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

#include "ops/op_base/op_python_call.h"
#include <cstddef>
#include <cstring>
#include "ops/utils/data_convert.h"
#include "hardware/hardware_abstract/collective/collective_manager.h"
#include "hardware/hardware_abstract/device_context.h"
#include "hardware/hardware_abstract/device_context_manager.h"
#include "ops/utils/async.h"

namespace mrt {
namespace ops {
namespace {
constexpr size_t kInputModuleNameIndex = 0;
constexpr size_t kInputFuncNameIndex = 1;
constexpr size_t kRealInputOffset = 2;
constexpr auto kTorchOpsModule = "torch.ops";
constexpr auto kTorchOpsInnerModule = "torch._ops";

/**
 * @brief Update ATen tensor metadata and data pointer without creating new Python object
 */
void UpdateAtenTensor(at::Tensor &atTensor, const ir::TensorPtr &mrtTensor) {
  CHECK_IF_NULL(mrtTensor);

  void *newDataPtr = const_cast<void *>(mrtTensor->DataPtr());
  void *oldDataPtr = atTensor.data_ptr();
  const auto &newShape = mrtTensor->Shape();
  const auto &oldShape = atTensor.sizes();

  if (newDataPtr == oldDataPtr && newShape.size() == oldShape.size() &&
      std::equal(newShape.begin(), newShape.end(), oldShape.begin())) {
    return;
  }

  auto tensor_impl = atTensor.unsafeGetTensorImpl();
  at::DataPtr dataPtr(newDataPtr, atTensor.device());
  atTensor.storage().unsafeGetStorageImpl()->set_data_ptr(std::move(dataPtr));

  auto strides = mrtTensor->Strides();
  if (strides.empty()) {
    tensor_impl->set_sizes_contiguous(newShape);
  } else {
    tensor_impl->set_sizes_and_strides(newShape, strides);
  }
}

py::function GetPythonCallable(const std::string &moduleName, const std::string &funcName) {
  py::gil_scoped_acquire gil;
  try {
    if (moduleName.rfind(kTorchOpsInnerModule, 0) != 0) {
      LOG_EXCEPTION << "Python callable only supports '" << kTorchOpsInnerModule << "' prefix, got: " << moduleName;
      return {};
    }

    std::string subModuleName = moduleName.substr(strlen(kTorchOpsInnerModule));
    if (!subModuleName.empty() && subModuleName.front() == '.') {
      subModuleName.erase(0, 1);
    }

    if (subModuleName.empty()) {
      LOG_EXCEPTION << "invalid moduleName: " << moduleName;
      return {};
    }

    py::module_ torchOps = py::module_::import(kTorchOpsModule);
    py::object subMod = torchOps.attr(subModuleName.c_str());
    py::object func = subMod.attr(funcName.c_str());
    if (func.is_none()) {
      LOG_EXCEPTION << "attribute '" + funcName + "' is None";
    }
    return func.cast<py::function>();
  } catch (const std::exception &e) {
    LOG_EXCEPTION << "Failed to get Python callable [" + moduleName + "." << funcName + "]: " << e.what();
    return {};
  }
}

void AttachAtenTensorCopy(const at::Tensor &atenTensor, ir::TensorPtr &irTensor, const std::string &logPrefix,
                          device::DeviceContext *devCtx) {
  std::vector<int64_t> atenShape(atenTensor.sizes().begin(), atenTensor.sizes().end());
  if (atenShape != irTensor->Shape()) {
    LOG_EXCEPTION << logPrefix << " shape mismatch, expect " << irTensor->Shape() << ", but got " << atenShape;
  }

  void *src = atenTensor.data_ptr();
  void *dst = irTensor->DataPtr();
  const size_t numBytes = irTensor->GetStorage()->SizeBytes();
  auto launchTask = [dst, src, numBytes, devCtx]() -> int {
    auto stream = devCtx->deviceResManager_->GetCurrentStream();
    if (!devCtx->deviceResManager_->AsyncCopy(dst, src, numBytes, device::CopyType::D2D, stream)) {
      LOG_EXCEPTION << "PythonCall device-to-device copy failed.";
      return UNKNOWN_ERROR;
    }
    return SUCCESS;
  };

  const auto &launchOpFunc = ops::OpAsync::GetLaunchOpFunc();
  if (launchOpFunc != nullptr) {
    launchOpFunc(logPrefix, launchTask, false);
  } else {
    (void)launchTask();
  }
}
}  // namespace

// Jump Table definition: None(0), Tensor(1), Double(2), Int(3), Bool(4), String(5), Tuple(6), Symbol(7)
const OpPythonCall::ConvertFunc OpPythonCall::kInputConverterTable[] = {
  &OpPythonCall::ConvertNoneToPy,    // Tag::None = 0
  &OpPythonCall::ConvertTensorToPy,  // Tag::Tensor = 1
  &OpPythonCall::ConvertDoubleToPy,  // Tag::Double = 2
  &OpPythonCall::ConvertIntToPy,     // Tag::Int = 3
  &OpPythonCall::ConvertBoolToPy,    // Tag::Bool = 4
  &OpPythonCall::ConvertStringToPy,  // Tag::String = 5
  &OpPythonCall::ConvertTupleToPy,   // Tag::Tuple = 6
  &OpPythonCall::ConvertIntToPy      // Tag::Symbol = 7 (treated as Int)
};

void OpPythonCall::Init(const std::vector<const ir::Value *> &inputs, const ir::Value *output) {
  LOG_OUT << "Input size: " << inputs.size();
  moduleName_ = inputs[kInputModuleNameIndex]->ToString();
  opName_ = inputs[kInputFuncNameIndex]->ToString();

  auto inputSize = inputs.size() - kRealInputOffset;
  inputs_.resize(inputSize, nullptr);
  inputTagIndices_.reserve(inputSize);

  for (size_t i = kRealInputOffset; i < inputs.size(); i++) {
    inputs_[i - kRealInputOffset] = inputs[i];
    inputTagIndices_.push_back(static_cast<size_t>(inputs[i]->GetTag()));
  }

  pyFunc_ = GetPythonCallable(moduleName_, opName_);
  auto deviceId = mrt::collective::CollectiveManager::Instance().local_rank_id();
  mrt::device::DeviceContextKey deviceContextKey = {hardware::GetDeviceNameByType(hardware::DeviceType::NPU), deviceId};
  dev_ctx_ = mrt::device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(deviceContextKey);
  CHECK_IF_NULL(dev_ctx_);
  CHECK_IF_NULL(dev_ctx_->deviceResManager_);

  // Reset state for new initialization
  firstRun_ = true;
  atTensors_.clear();
  tensorIdx_ = 0;
}

OpsErrorCode OpPythonCall::Launch(const std::vector<const ir::Value *> &input, void *workspace, size_t workspaceSize,
                                  ir::Value *output, void *stream) {
  return SUCCESS;
}

bool OpPythonCall::NeedLaunch() { return false; }

py::object OpPythonCall::ConvertNoneToPy(const ir::Value *value) { return py::none(); }

py::object OpPythonCall::ConvertTensorToPy(const ir::Value *value) {
  if (firstRun_) {
    auto atTensor = ToTorchTensor(value->ToTensor());
    atTensors_.push_back(atTensor);
    tensorIdx_++;
    return py::cast(atTensor);
  } else {
    CHECK_IF_FAIL(tensorIdx_ < atTensors_.size());
    auto &atTensor = atTensors_[tensorIdx_];
    UpdateAtenTensor(atTensor, value->ToTensor());
    tensorIdx_++;
    return py::cast(atTensor);
  }
}

py::object OpPythonCall::ConvertIntToPy(const ir::Value *value) { return py::cast(value->ToInt()); }

py::object OpPythonCall::ConvertDoubleToPy(const ir::Value *value) { return py::cast(value->ToDouble()); }

py::object OpPythonCall::ConvertBoolToPy(const ir::Value *value) { return py::cast(value->ToBool()); }

py::object OpPythonCall::ConvertStringToPy(const ir::Value *value) { return py::cast(value->ToString()); }

py::object OpPythonCall::ConvertTupleToPy(const ir::Value *value) {
  const auto &tuple = value->ToTuple();
  py::list pyList(tuple->Size());
  for (size_t i = 0; i < tuple->Size(); ++i) {
    auto *elem = tuple->operator[](i).get();
    auto tagIdx = static_cast<size_t>(elem->GetTag());
    if (tagIdx < kConverterCount) {
      pyList[i] = (this->*kInputConverterTable[tagIdx])(elem);
    } else {
      LOG_EXCEPTION << "Invalid tuple element tag: " << static_cast<int>(elem->GetTag());
    }
  }
  return std::move(pyList);
}

OpsErrorCode OpPythonCall::CalcWorkspace(const std::vector<const ir::Value *> &input, const ir::Value *output,
                                         size_t *workspaceSize) {
  if (Py_IsInitialized() == 0) {
    LOG_EXCEPTION << "Python interpreter is not initialized.";
    return UNKNOWN_ERROR;
  }

  py::gil_scoped_acquire gilAcquire;
  tensorIdx_ = 0;

  // Use Jump Table for O(1) dispatch based on cached type indices
  py::tuple pyArgs(inputs_.size());
  for (size_t i = 0; i < inputs_.size(); i++) {
    auto tagIdx = inputTagIndices_[i];
    if (tagIdx < kConverterCount) {
      pyArgs[i] = (this->*kInputConverterTable[tagIdx])(inputs_[i]);
    } else {
      LOG_EXCEPTION << "Invalid input tag: " << tagIdx << " at index " << i;
    }
  }

  LOG_OUT << "input size: " << pyArgs.size();

  if (pyFunc_.is_none()) {
    LOG_EXCEPTION << "Python function object is null, func name: " << opName_;
    return UNKNOWN_ERROR;
  }

  py::object result;
  try {
    result = inputs_.empty() ? pyFunc_() : pyFunc_(*pyArgs);
  } catch (const std::exception &e) {
    LOG_EXCEPTION << "Python function call failed: " << e.what();
    return UNKNOWN_ERROR;
  }

  auto ret = PostprocessOutputs(result, const_cast<ir::Value *>(output));
  CheckOutputInputRef(inputs_, output, opName_);
  firstRun_ = false;
  return ret;
}

OpsErrorCode OpPythonCall::PostprocessOutputs(py::handle result, ir::Value *output) {
  if (!output || output->IsNone()) {
    LOG_OUT << "PythonCall op " << opName_ << " has no output tensor; ";
    return SUCCESS;
  }

  if (py::isinstance<py::tuple>(result)) {
    if (!output->IsTuple()) {
      LOG_EXCEPTION << "PythonCall op " << opName_ << " expects tuple but IR output is not tuple.";
    }
    py::tuple tup = result.cast<py::tuple>();
    auto irList = output->ToTuple()->ToTensorList();
    if (irList.size() != tup.size()) {
      LOG_EXCEPTION << "PythonCall op " << opName_ << " tuple size mismatch: expect " << irList.size() << ", got "
                    << tup.size();
    }
    for (size_t i = 0; i < tup.size(); ++i) {
      at::Tensor aten;
      try {
        aten = tup[i].cast<at::Tensor>();
      } catch (...) {
        LOG_EXCEPTION << "PythonCall op " << opName_ << " tuple[" << i << "] is not a torch.Tensor";
      }
      AttachAtenTensorCopy(aten, irList[i], "PythonCall op " + opName_ + " tuple[" + std::to_string(i) + "]", dev_ctx_);
    }
    LOG_OUT << "PythonCall op " << opName_ << " zero-copy attached " << tup.size() << " tensors into output tuple.";
    return SUCCESS;
  }

  at::Tensor aten;
  try {
    aten = result.cast<at::Tensor>();
  } catch (...) {
    LOG_EXCEPTION << "PythonCall op " << opName_ << " result is not a torch.Tensor";
  }
  ir::TensorPtr irTensor = output->ToTensor();
  if (!irTensor) {
    LOG_EXCEPTION << "PythonCall op " << opName_ << " output tensor pointer is null.";
  }
  AttachAtenTensorCopy(aten, irTensor, "PythonCall op " + opName_, dev_ctx_);
  LOG_OUT << "PythonCall op " << opName_ << " zero-copy attached single at::Tensor data ptr.";
  return SUCCESS;
}

}  // namespace ops
}  // namespace mrt
