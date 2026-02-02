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

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include "config/device/ascend/op_precision_conf.h"

namespace nb = nanobind;
using OpPrecisionConf = mrt::config::ascend::OpPrecisionConf;

NB_MODULE(_mrt_config, m) {
  m.doc() = "Python binding for MRT OpPrecisionConf";
  (void)nb::class_<OpPrecisionConf>(m, "OpPrecisionConf")
    .def_static("Instance", &OpPrecisionConf::Instance, nb::rv_policy::reference)
    .def("set_is_allow_matmul_hf32", &OpPrecisionConf::SetIsAllowMatmulHF32)
    .def("set_acl_precision_mode", &OpPrecisionConf::SetAclPrecisionMode);
}
