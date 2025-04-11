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

// Interface with python
PYBIND11_MODULE(_dalang, mod) {
  (void)py::class_<DALangPy, std::shared_ptr<DALangPy>>(mod, "DALangPy_")
      .def_static("get_instance", &DALangPy::GetInstance,
                  "DALangPy single instance.")
      .def("__call__", &DALangPy::Run, py::arg("args"), "Run with arguments.")
      .def("compile", &DALangPy::Compile, py::arg("source"),
           py::arg("args") = py::list(), "Compile the source with arguments.");
}
