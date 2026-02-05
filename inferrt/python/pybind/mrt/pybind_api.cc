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

#include <nanobind/stl/string.h>
#include <nanobind/stl/shared_ptr.h>

#include "mrt/pybind_api.h"
#include "ops/custom_op_register.h"

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

std::vector<Argument> DALangPy::ConvertPyArgs(const nb::tuple &args) {
  std::vector<Argument> arguments;
  for (const auto &arg : args) {
    if (nb::isinstance<nb::bool_>(arg)) {
      Argument argument = Argument({.type = da::vm::SlotBool});
      argument.value.bool_ = nb::cast<bool>(arg);
      arguments.emplace_back(std::move(argument));
    } else if (nb::isinstance<nb::int_>(arg)) {
      Argument argument = Argument({.type = da::vm::SlotInt});
      argument.value.int_ = nb::cast<int>(arg);
      arguments.emplace_back(std::move(argument));
    } else if (nb::isinstance<nb::float_>(arg)) {
      Argument argument = Argument({.type = da::vm::SlotFloat});
      argument.value.float_ = nb::cast<double>(arg);
      arguments.emplace_back(std::move(argument));
    } else if (nb::isinstance<nb::str>(arg)) {
      Argument argument = Argument({.type = da::vm::SlotString});
      // Potential dangling pointer if Argument stores char* and not std::string
      // Preserving original logic structure
      argument.value.str_ = nb::cast<std::string>(arg).c_str();
      arguments.emplace_back(std::move(argument));
    } else {
      Argument argument = Argument({.type = da::vm::SlotTensor});
      // get tensor data from ms tensor
      auto tensor = callable_->graphExecutor().AddInputNode();
      argument.tensor_ = tensor;
      arguments.emplace_back(std::move(argument));
    }
  }
  return arguments;
}

nb::object DALangPy::ConvertPyResult(const Result &res) {
  if (res.type == da::vm::SlotBool) {
    return nb::bool_(res.value.bool_);
  } else if (res.type == da::vm::SlotInt) {
    return nb::int_(res.value.int_);
  } else if (res.type == da::vm::SlotFloat) {
    return nb::float_(res.value.float_);
  } else if (res.type == da::vm::SlotString) {
    return nb::str(res.value.str_);
  } else if (res.type == da::vm::SlotTensor) {
    return nb::object();
  } else if (res.type == da::vm::SlotVoid) {
    return nb::none();
  } else {
    return nb::none();
  }
}

void DALangPy::Compile(const nb::object &source, bool graph, bool dump) {
  // Check if the function or net is valid.
  if ((!nb::isinstance<nb::str>(source))) {
    LOG_ERROR << "error: the source must be string.";
    exit(EXIT_FAILURE);
  }
  const auto srcStr = nb::cast<std::string>(source);
  callable_ = DA_API_Compile(srcStr.c_str(), graph, dump);
}

nb::object DALangPy::Run(const nb::tuple &args) {
  CHECK_IF_NULL(callable_);
  auto res = DA_API_Run(callable_, ConvertPyArgs(args));
  auto pyres = ConvertPyResult(res);
  if (pyres.ptr() == nullptr) {
    return nb::none();
  }
  LOG_OUT << "res: " << da::vm::ToString(res) << ", pyres: " << nb::cast<std::string>(nb::str(pyres));
  return pyres;
}

// Interface with python
NB_MODULE(_mrt_api, mod) {
  (void)nb::class_<DALangPy>(mod, "DALangPy_")
    .def_static("get_instance", &DALangPy::GetInstance, "DALangPy single instance.")
    .def("__call__", &DALangPy::Run, nb::arg("args") = nb::list(), "Run with arguments.")
    .def("compile", &DALangPy::Compile, nb::arg("source"), nb::arg("graph") = nb::bool_(false),
         nb::arg("dump") = nb::bool_(false), "Compile the source with arguments.");

  mod.def(
    "is_custom_op_registered",
    [](const std::string &op_name) { return mrt::ops::CustomOpRegistry::GetInstance().IsCustomOpRegistered(op_name); },
    nb::arg("op_name"), "Check if a custom operator is registered.");
}
