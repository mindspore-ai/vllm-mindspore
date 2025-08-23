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

#include "pybind/dalang_py/pybind11_api.h"

DALangPy::~DALangPy() {
  if (callable_ != nullptr) {
    delete callable_;
    callable_ = nullptr;
  }
}

std::shared_ptr<DALangPy> DALangPy::GetInstance() {
  static auto dalangPy = std::make_shared<DALangPy>();
  return dalangPy;
}

std::vector<Argument> DALangPy::ConvertPyArgs(const py::tuple &args) {
  std::vector<Argument> arguments;
  for (const auto &arg : args) {
    if (py::isinstance<py::bool_>(arg)) {
      Argument argument = Argument({.type = da::vm::SlotBool});
      argument.value.bool_ = py::cast<bool>(arg);
      arguments.emplace_back(std::move(argument));
    } else if (py::isinstance<py::int_>(arg)) {
      Argument argument = Argument({.type = da::vm::SlotInt});
      argument.value.int_ = py::cast<int>(arg);
      arguments.emplace_back(std::move(argument));
    } else if (py::isinstance<py::float_>(arg)) {
      Argument argument = Argument({.type = da::vm::SlotFloat});
      argument.value.float_ = py::cast<double>(arg);
      arguments.emplace_back(std::move(argument));
    } else if (py::isinstance<py::str>(arg)) {
      Argument argument = Argument({.type = da::vm::SlotString});
      argument.value.str_ = py::cast<std::string>(arg).c_str();
      arguments.emplace_back(std::move(argument));
    } else {
      Argument argument = Argument({.type = da::vm::SlotTensor});
      // get tensor data from ms tensor
      void *data = (void *)arg.ptr();
      auto tensor = callable_->graphExecutor().AddTensor(da::tensor::Type_F32, 0, {0}, data);
      argument.value.tensor_ = (void *)tensor;
      arguments.emplace_back(std::move(argument));
    }
  }
  return arguments;
}

py::object DALangPy::ConvertPyResult(const Result &res) {
  if (res.type == da::vm::SlotBool) {
    return py::bool_(res.value.bool_);
  } else if (res.type == da::vm::SlotInt) {
    return py::int_(res.value.int_);
  } else if (res.type == da::vm::SlotFloat) {
    return py::float_(res.value.float_);
  } else if (res.type == da::vm::SlotString) {
    return py::str(res.value.str_);
  } else if (res.type == da::vm::SlotTensor) {
    return py::object();
  } else if (res.type == da::vm::SlotVoid) {
    return py::none();
  } else {
    return py::none();
  }
}

void DALangPy::Compile(const py::object &source, bool graph, bool dump) {
  // Check if the function or net is valid.
  if ((!py::isinstance<py::str>(source))) {
    LOG_ERROR << "error: the source must be string.";
    exit(EXIT_FAILURE);
  }
  const auto srcStr = py::cast<std::string>(source);
  callable_ = DA_API_Compile(srcStr.c_str(), graph, dump);
}

py::object DALangPy::Run(const py::tuple &args) {
  CHECK_IF_NULL(callable_);
  auto res = DA_API_Run(callable_, ConvertPyArgs(args));
  auto pyres = ConvertPyResult(res);
  if (pyres.ptr() == nullptr) {
    return py::none();
  }
  LOG_OUT << "res: " << da::vm::ToString(res) << ", pyres: " << py::str(pyres).cast<std::string>();
  return pyres;
}

// Interface with python
PYBIND11_MODULE(_dapy, mod) {
  (void)py::class_<DALangPy, std::shared_ptr<DALangPy>>(mod, "DALangPy_")
    .def_static("get_instance", &DALangPy::GetInstance, "DALangPy single instance.")
    .def("__call__", &DALangPy::Run, py::arg("args") = py::list(), "Run with arguments.")
    .def("compile", &DALangPy::Compile, py::arg("source"), py::arg("graph") = py::bool_(false),
         py::arg("dump") = py::bool_(false), "Compile the source with arguments.");
}
