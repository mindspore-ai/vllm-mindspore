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
#include <nanobind/stl/vector.h>
#include <nanobind/stl/pair.h>

#include "config/device/ascend/aclgraph_conf.h"

namespace nb = nanobind;
using AclGraphConf = mrt::config::ascend::AclGraphConf;

NB_MODULE(_aclgraph_config, m) {
  m.doc() = "Python binding for MRT AclGraphConf";
  nb::set_leak_warnings(false);
  (void)nb::class_<AclGraphConf>(m, "AclGraphConf")
    .def_static("Instance", &AclGraphConf::Instance, nb::rv_policy::reference)
    .def("set_pool_id", &AclGraphConf::SetPoolId)
    .def("pool_id", &AclGraphConf::GetPoolId)
    .def("set_op_capture_skip", &AclGraphConf::SetOpCaptureSkip)
    .def("begin_capture", &AclGraphConf::BeginCapture)
    .def("end_capture", &AclGraphConf::EndCapture);
}
