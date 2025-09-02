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

#include "hardware/hardware_abstract/collective/communication_group.h"

namespace mrt {
namespace collective {
CommunicationGroup::CommunicationGroup(const std::string &name, const std::vector<uint32_t> &groupRanks,
                                       uint32_t groupRank, int64_t comm)
    : groupName(name), groupRanks(groupRanks), groupRank(groupRank), comm(comm) {}

const std::string &CommunicationGroup::group_name() const { return groupName; }

const std::vector<uint32_t> &CommunicationGroup::group_ranks() const { return groupRanks; }

uint32_t CommunicationGroup::group_size() const { return groupRanks.size(); }

uint32_t CommunicationGroup::group_rank() const { return groupRank; }

int64_t CommunicationGroup::communicator() const { return comm; }

}  // namespace collective
}  // namespace mrt
