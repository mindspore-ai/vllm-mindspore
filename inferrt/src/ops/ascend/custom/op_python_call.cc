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
      py::list py_list(tuple->Size());
      for (size_t i = 0; i < tuple->Size(); ++i) {
        py_list[i] = ValueToPyData(tuple->operator[](i).get());
      }
      return py_list;
    }
    if (input->IsSymbol()) {
      return py::cast(input->ToInt());
    }
  } catch (const std::exception &e) {
    LOG_EXCEPTION << "Failed to convert Value to Python object: " << e.what();
  }
  return py::none();
}

py::function GetPythonCallable(const std::string &module_name, const std::string &func_name) {
  py::gil_scoped_acquire gil;
  try {
    if (module_name.rfind(kTorchOpsInnerModule, 0) != 0) {
      LOG_EXCEPTION << "Python callable only supports '" << kTorchOpsInnerModule << "' prefix, got: " << module_name;
      return {};
    }

    std::string sub_module_name = module_name.substr(strlen(kTorchOpsInnerModule));
    if (!sub_module_name.empty() && sub_module_name.front() == '.') {
      sub_module_name.erase(0, 1);
    }

    if (sub_module_name.empty()) {
      LOG_EXCEPTION << "invalid module_name: " << module_name;
      return {};
    }

    py::module_ torch_ops = py::module_::import(kTorchOpsModule);
    py::object sub_mod = torch_ops.attr(sub_module_name.c_str());
    py::object func = sub_mod.attr(func_name.c_str());
    if (func.is_none()) {
      LOG_EXCEPTION << "attribute '" + func_name + "' is None";
    }
    return func.cast<py::function>();
  } catch (const std::exception &e) {
    LOG_EXCEPTION << "Failed to get Python callable [" + module_name + "." << func_name + "]: " + e.what();
    return {};
  }
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

OpsErrorCode OpPythonCall::InferShape(const std::vector<const ir::Value *> &input, ir::Value *output) {
  Operator::InferShape(input, output);
  return SUCCESS;
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
  py::gil_scoped_acquire gil_acquire;
  py::tuple py_args = PreprocessInputs(inputs_);
  py::object result;
  LOG_OUT << "input size: " << py_args.size();

  if (pyFunc_.is_none()) {
    LOG_EXCEPTION << "Python function object is null, func name: " << opName_;
    return UNKNOWN_ERROR;
  }

  try {
    result = inputs_.empty() ? pyFunc_() : pyFunc_(*py_args);
  } catch (const std::exception &e) {
    LOG_EXCEPTION << "Python function call failed: " << e.what();
    return UNKNOWN_ERROR;
  }
  (void)c10_npu::getCurrentNPUStream().stream(true);
  return PostprocessOutputs(result, output, stream);
}

py::tuple OpPythonCall::PreprocessInputs(const std::vector<const ir::Value *> &input) {
  pybind11::tuple py_args(input.size());
  for (size_t i = 0; i < input.size(); i++) {
    py_args[i] = ValueToPyData(input[i]);
  }

  return py_args;
}

OpsErrorCode OpPythonCall::PostprocessOutputs(py::handle result, ir::Value *output, void *stream) {
  if (!output) {
    LOG_OUT << "PythonCall op " << opName_ << " has no output tensor; ";
    return SUCCESS;
  }

  if (!output->IsTensor()) {
    LOG_EXCEPTION << "PythonCall op " << opName_ << " output value is not a tensor.";
    return UNKNOWN_ERROR;
  }

  ir::TensorPtr ir_tensor = output->ToTensor();
  if (!ir_tensor) {
    LOG_EXCEPTION << "PythonCall op " << opName_ << " output tensor pointer is null.";
    return UNKNOWN_ERROR;
  }

  py::module_ torch = py::module_::import("torch");
  if (!torch.attr("is_tensor")(result).cast<bool>()) {
    LOG_EXCEPTION << "result  is not a torch.Tensor";
  }

  void *src_ptr = reinterpret_cast<void *>(result.attr("data_ptr")().cast<uintptr_t>());
  const size_t actual_bytes = result.attr("nbytes").cast<size_t>();
  const size_t expect_bytes = ir_tensor->GetStorage()->SizeBytes();

  if (actual_bytes != expect_bytes) {
    LOG_EXCEPTION << "PythonCall op " << opName_ << " byte-size mismatch: python tensor " << actual_bytes
                  << " vs ir tensor " << expect_bytes << ".";
    return UNKNOWN_ERROR;
  }

  if (!mrt::device::ascend::AscendResManager::MemcpyDeviceToDevice(ir_tensor->DataPtr(), expect_bytes, src_ptr,
                                                                   expect_bytes, stream)) {
    LOG_EXCEPTION << "PythonCall op " << opName_ << " device-to-device copy failed.";
    return UNKNOWN_ERROR;
  }

  LOG_OUT << "PythonCall op " << opName_ << " copied " << expect_bytes << " bytes to output tensor successfully.";
  return SUCCESS;
}

MRT_REG_OP(python_call, OpPythonCall, Ascend);

}  // namespace ops
}  // namespace mrt
