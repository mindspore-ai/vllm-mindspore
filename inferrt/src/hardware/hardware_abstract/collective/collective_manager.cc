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

#include "hardware/hardware_abstract/collective/collective_manager.h"

namespace mrt {
namespace collective {

CollectiveManager::CollectiveManager() : globalRankId(0), localRankId(0), globalRankSize(0) {}

CollectiveManager::~CollectiveManager() {}

CollectiveManager &CollectiveManager::Instance() {
  static CollectiveManager instance;
  return instance;
}

bool CollectiveManager::CreateCommunicationGroup(const std::string &groupName, const std::vector<uint32_t> &groupRanks,
                                                 uint32_t groupRank, int64_t communicator) {
  if (communicationGroups.find(groupName) != communicationGroups.end()) {
    return false;
  }
  communicationGroups[groupName] = std::make_shared<CommunicationGroup>(groupName, groupRanks, groupRank, communicator);
  return true;
}

bool CollectiveManager::IsGroupExist(const std::string &groupName) {
  return communicationGroups.find(groupName) != communicationGroups.end();
}

std::shared_ptr<CommunicationGroup> CollectiveManager::GetCommunicationGroup(const std::string &groupName) {
  if (communicationGroups.find(groupName) == communicationGroups.end()) {
    LOG_EXCEPTION << "can not find group for given group name " << groupName;
    return nullptr;
  }
  return communicationGroups[groupName];
}

uint32_t CollectiveManager::GetGroupRank(const std::string &groupName) {
  if (communicationGroups.find(groupName) == communicationGroups.end()) {
    LOG_ERROR << "can not find group for given group name " << groupName;
    return false;
  }
  return communicationGroups[groupName]->group_rank();
}

uint32_t CollectiveManager::GetGroupSize(const std::string &groupName) {
  if (communicationGroups.find(groupName) == communicationGroups.end()) {
    LOG_ERROR << "can not find group for given group name " << groupName;
    return false;
  }
  return communicationGroups[groupName]->group_size();
}

void CollectiveManager::SetGlobalRankId(uint32_t globalRankId) { this->globalRankId = globalRankId; }

void CollectiveManager::SetGlobalRankSize(uint32_t globalRankSize) { this->globalRankSize = globalRankSize; }

void CollectiveManager::SetLocalRankId(uint32_t localRankId) { this->localRankId = localRankId; }

uint32_t CollectiveManager::global_rank_id() const { return globalRankId; }

uint32_t CollectiveManager::local_rank_id() const { return localRankId; }

uint32_t CollectiveManager::global_rank_size() const { return globalRankSize; }

}  // namespace collective
}  // namespace mrt
