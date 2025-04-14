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

std::shared_ptr<DALangPy> DALangPy::GetInstance() {
  static auto dalangPy = std::make_shared<DALangPy>();
  return dalangPy;
}

void DALangPy::Compile(const py::object &source, const py::tuple &args,
                       bool dump) {
  // Check if the function or net is valid.
  if ((!py::isinstance<py::str>(source))) {
    LOG_ERROR << "error: the source must be string.";
    exit(EXIT_FAILURE);
  }
  auto srcStr = py::cast<std::string>(source);
  LOG_OUT << "source:\n" << srcStr;
  constexpr auto pythonDef = "def";
  const char *functionStr = strstr(srcStr.c_str(), pythonDef);
  if (functionStr == nullptr) {
    LOG_ERROR << "error: keyword 'def' not found.";
    exit(EXIT_FAILURE);
  }

  auto lexer = lexer::Lexer(functionStr);
  if (dump) {
    lexer.Dump();
  }

  auto parser = parser::Parser(&lexer);
  parser.ParseCode();
  if (dump) {
    parser.DumpAst();
  }

  auto compiler = new compiler::Compiler(&parser); // delete by user.
  compiler->Compile();
  if (dump) {
    compiler->Dump();
  }

  LOG_OUT << "Save callable: " << compiler;
  callable_ = compiler;
}

void DALangPy::Run(const py::tuple &args) {
  CHECK_NULL(callable_);
  LOG_OUT << "Run callable: " << callable_;
  auto vm = vm::VM(callable_);
  vm.Run();
}

// Interface with python
PYBIND11_MODULE(_dapy, mod) {
  (void)py::class_<DALangPy, std::shared_ptr<DALangPy>>(mod, "DALangPy_")
      .def_static("get_instance", &DALangPy::GetInstance,
                  "DALangPy single instance.")
      .def("__call__", &DALangPy::Run, py::arg("args"), "Run with arguments.")
      .def("compile", &DALangPy::Compile, py::arg("source"),
           py::arg("args") = py::list(), py::arg("dump") = py::bool_(false),
           "Compile the source with arguments.");
}
