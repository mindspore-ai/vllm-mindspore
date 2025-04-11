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

#ifndef __DALANG_PYBIND11_API_H__
#define __DALANG_PYBIND11_API_H__

#include "pybind11/pybind11.h"

#include "../../dalang/api/c_api.h"

namespace py = pybind11;
using Callable = compiler::Compiler;

class DALangPy {
public:
  DALangPy() = default;
  ~DALangPy() = default;
  DALangPy(const DALangPy &) = delete;
  DALangPy(DALangPy &&) = delete;
  DALangPy &operator=(const DALangPy &) = delete;
  DALangPy &operator=(DALangPy &&) = delete;

  static DALangPy *GetInstance();

  void Compile(const py::object &source, const py::tuple &args);
  void Run(const py::tuple &args);

private:
  Callable *callable_{nullptr};
};

#endif // __DALANG_PYBIND11_API_H__