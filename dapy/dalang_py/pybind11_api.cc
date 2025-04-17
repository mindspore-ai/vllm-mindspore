/**
 * Copyright 2025 Zhang Qinghua
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

#include "pybind11_api.h"

#undef DEBUG
#ifndef DEBUG
#undef LOG_OUT
#define LOG_OUT NO_LOG_OUT
#endif

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

namespace {
std::vector<Argument> ConvertPyArgs(const py::tuple &args) {
  std::vector<Argument> arguments;
  for (const auto &arg : args) {
    if (py::isinstance<py::bool_>(arg)) {
      Argument argument = Argument({.type = vm::SlotBool});
      argument.value.bool_ = py::cast<bool>(arg);
      arguments.emplace_back(std::move(argument));
    } else if (py::isinstance<py::int_>(arg)) {
      Argument argument = Argument({.type = vm::SlotInt});
      argument.value.int_ = py::cast<int>(arg);
      arguments.emplace_back(std::move(argument));
    } else if (py::isinstance<py::float_>(arg)) {
      Argument argument = Argument({.type = vm::SlotFloat});
      argument.value.float_ = py::cast<double>(arg);
      arguments.emplace_back(std::move(argument));
    } else if (py::isinstance<py::str>(arg)) {
      Argument argument = Argument({.type = vm::SlotString});
      argument.value.str_ = py::cast<std::string>(arg).c_str();
      arguments.emplace_back(std::move(argument));
    } else {
      Argument argument = Argument({.type = vm::SlotTensor});
      argument.value.tensor_ = (void *)py::cast<Tensor *>(arg);
      arguments.emplace_back(std::move(argument));
    }
  }
  return arguments;
}

py::object ConvertPyResult(const Result &res) {
  if (res.type == vm::SlotBool) {
    return py::bool_(res.value.bool_);
  } else if (res.type == vm::SlotInt) {
    return py::int_(res.value.int_);
  } else if (res.type == vm::SlotFloat) {
    return py::float_(res.value.float_);
  } else if (res.type == vm::SlotString) {
    return py::str(res.value.str_);
  } else if (res.type == vm::SlotTensor) {
    return py::object();
  } else if (res.type == vm::SlotVoid) {
    return py::none();
  } else {
    return py::none();
  }
}
} // namespace

void DALangPy::Compile(const py::object &source, bool dump) {
  // Check if the function or net is valid.
  if ((!py::isinstance<py::str>(source))) {
    LOG_ERROR << "error: the source must be string.";
    exit(EXIT_FAILURE);
  }
  const auto srcStr = py::cast<std::string>(source);
  callable_ = DA_API_Compile(srcStr.c_str(), dump);
}

py::object DALangPy::Run(const py::tuple &args) {
  CHECK_IF_NULL(callable_);
  auto res = DA_API_Run(callable_, ConvertPyArgs(args));
  LOG_OUT << "result: " << vm::ToString(res);
  return ConvertPyResult(res);
}

// Interface with python
PYBIND11_MODULE(_dapy, mod) {
  (void)py::class_<DALangPy, std::shared_ptr<DALangPy>>(mod, "DALangPy_")
      .def_static("get_instance", &DALangPy::GetInstance,
                  "DALangPy single instance.")
      .def("__call__", &DALangPy::Run, py::arg("args") = py::list(),
           "Run with arguments.")
      .def("compile", &DALangPy::Compile, py::arg("source"),
           py::arg("dump") = py::bool_(false),
           "Compile the source with arguments.");
}
