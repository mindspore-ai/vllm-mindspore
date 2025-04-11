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

namespace py = pybind11;

class DALangPy {
public:
  DALangPy() = default;
  ~DALangPy() = default;

  static DALangPy *GetInstance() {
    static auto dalangPy = DALangPy();
    return &dalangPy;
  }

  void Compile(const py::object &source, const py::tuple &args) {}
  void Run(const py::tuple &args) {}
};

#endif // __DALANG_PYBIND11_API_H__