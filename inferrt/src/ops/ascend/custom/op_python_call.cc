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

#include <pybind11/functional.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include "ops/ascend/custom/op_python_call.h"
#include "ops/utils/data_convert.h"
#include "ops/custom_op_register.h"
#include "hardware/ascend/res_manager/ascend_res_manager.h"

namespace mrt {
namespace ops {
namespace {
constexpr size_t kInputModuleNameIndex = 0;
constexpr size_t kInputFuncNameIndex = 1;
constexpr size_t kRealInputOffset = 2;
constexpr auto kTorchOpsModule = "torch.ops";
constexpr auto kTorchOpsInnerModule = "torch._ops";

py::object ValueToPyData(const ir::Value *input) {
  try {
    if (input == nullptr || input->IsNone()) {
      return py::none();
    }
    if (input->IsTensor()) {
      return py::cast(ToTorchTensor(input->ToTensor()));
    }
    if (input->IsDouble()) {
      return py::cast(input->ToDouble());
    }
    if (input->IsInt()) {
      return py::cast(input->ToInt());
    }
    if (input->IsBool()) {
      return py::cast(input->ToBool());
    }
    if (input->IsString()) {
      return py::cast(input->ToString());
    }
    if (input->IsTuple()) {
      const auto &tuple = input->ToTuple();
      py::list pyList(tuple->Size());
      for (size_t i = 0; i < tuple->Size(); ++i) {
        pyList[i] = ValueToPyData(tuple->operator[](i).get());
      }
      return pyList;
    }
    if (input->IsSymbol()) {
      return py::cast(input->ToInt());
    }
  } catch (const std::exception &e) {
    LOG_EXCEPTION << "Failed to convert Value to Python object: " << e.what();
  }
  return py::none();
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
    LOG_EXCEPTION << "Failed to get Python callable [" + moduleName + "." << funcName + "]: " + e.what();
    return {};
  }
}

void AttachAtenTensorNoCopy(const at::Tensor &atenTensor, ir::TensorPtr &irTensor, const std::string &logPrefix) {
  std::vector<int64_t> atenShape(atenTensor.sizes().begin(), atenTensor.sizes().end());
  if (atenShape != irTensor->Shape()) {
    LOG_EXCEPTION << logPrefix << " shape mismatch, expect " << irTensor->Shape() << ", but got " << atenShape;
  }

  c10::DataPtr dataPtr = atenTensor.storage().set_data_ptr(std::move(c10::DataPtr()));
  void *data = dataPtr.get();

  auto dataPtrShared = std::make_shared<c10::DataPtr>(std::move(dataPtr));
  auto deleter = [dataPtrShared](void *) mutable {
    if (dataPtrShared) dataPtrShared->clear();
  };
  irTensor->GetStorage()->SetDataPtrFromAten(data, deleter);
}
}  // namespace

void OpPythonCall::Init(const std::vector<const ir::Value *> &inputs, const ir::Value *output) {
  LOG_OUT << "Input size: " << inputs.size();
  moduleName_ = inputs[kInputModuleNameIndex]->ToString();
  opName_ = inputs[kInputFuncNameIndex]->ToString();
  auto inputSize = inputs.size() - kRealInputOffset;
  inputs_.resize(inputSize, nullptr);
  for (size_t i = kRealInputOffset; i < inputs.size(); i++) {
    inputs_[i - kRealInputOffset] = inputs[i];
  }
  pyFunc_ = GetPythonCallable(moduleName_, opName_);
}

OpsErrorCode OpPythonCall::CalcWorkspace(const std::vector<const ir::Value *> &input, const ir::Value *output,
                                         size_t *workspaceSize) {
  return SUCCESS;
}

OpsErrorCode OpPythonCall::Launch(const std::vector<const ir::Value *> &input, void *workspace, size_t workspaceSize,
                                  ir::Value *output, void *stream) {
  if (Py_IsInitialized() == 0) {
    LOG_EXCEPTION << "Python interpreter is not initialized.";
    return UNKNOWN_ERROR;
  }
  py::gil_scoped_acquire gilAcquire;
  py::tuple pyArgs = PreprocessInputs(inputs_);
  py::object result;
  LOG_OUT << "input size: " << pyArgs.size();

  if (pyFunc_.is_none()) {
    LOG_EXCEPTION << "Python function object is null, func name: " << opName_;
    return UNKNOWN_ERROR;
  }

  try {
    result = inputs_.empty() ? pyFunc_() : pyFunc_(*pyArgs);
  } catch (const std::exception &e) {
    LOG_EXCEPTION << "Python function call failed: " << e.what();
    return UNKNOWN_ERROR;
  }
  (void)c10_npu::getCurrentNPUStream().stream(true);
  return PostprocessOutputs(result, output, stream);
}

py::tuple OpPythonCall::PreprocessInputs(const std::vector<const ir::Value *> &input) {
  pybind11::tuple pyArgs(input.size());
  for (size_t i = 0; i < input.size(); i++) {
    pyArgs[i] = ValueToPyData(input[i]);
  }

  return pyArgs;
}

OpsErrorCode OpPythonCall::PostprocessOutputs(py::handle result, ir::Value *output, void * /*stream*/) {
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
      AttachAtenTensorNoCopy(aten, irList[i], "PythonCall op " + opName_ + " tuple[" + std::to_string(i) + "]");
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
  AttachAtenTensorNoCopy(aten, irTensor, "PythonCall op " + opName_);
  LOG_OUT << "PythonCall op " << opName_ << " zero-copy attached single at::Tensor data ptr.";
  return SUCCESS;
}

MRT_REG_OP(python_call, OpPythonCall, Ascend);
MRT_REG_OP(python_call, OpPythonCall, CPU);
}  // namespace ops
}  // namespace mrt
