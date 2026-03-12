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
#include <sstream>

#include "hardware/hardware_abstract/collective/collective_manager.h"

namespace nb = nanobind;
using CollectiveManager = mrt::collective::CollectiveManager;

NB_MODULE(_ms_inferrt_collective, mod) {
  (void)nb::class_<CollectiveManager>(mod, "CollectiveManager")
    .def_static("instance", &CollectiveManager::Instance, nb::rv_policy::reference)
    .def("create_communication_group", &CollectiveManager::CreateCommunicationGroup)
    .def("is_group_exist", &CollectiveManager::IsGroupExist)
    .def("get_group_rank", &CollectiveManager::GetGroupRank)
    .def("get_group_size", &CollectiveManager::GetGroupSize)
    .def("set_global_rank_id", &CollectiveManager::SetGlobalRankId)
    .def("set_local_rank_id", &CollectiveManager::SetLocalRankId)
    .def("set_global_rank_size", &CollectiveManager::SetGlobalRankSize)
    .def("global_rank_id", &CollectiveManager::global_rank_id)
    .def("local_rank_id", &CollectiveManager::local_rank_id)
    .def("global_rank_size", &CollectiveManager::global_rank_size);
}
