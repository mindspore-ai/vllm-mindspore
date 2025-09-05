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

#ifndef INFERRT_SRC_HARDWARE_COMMON_H__
#define INFERRT_SRC_HARDWARE_COMMON_H__

#include "pybind11/pybind11.h"

namespace py = pybind11;
namespace mrt {
class GilReleaseWithCheck {
 public:
  GilReleaseWithCheck();

  ~GilReleaseWithCheck();

 private:
  std::unique_ptr<py::gil_scoped_release> release_;
};
}  // namespace mrt

#endif  // INFERRT_SRC_HARDWARE_COMMON_H__
