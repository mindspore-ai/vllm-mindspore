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
#ifndef INFERRT_SRC_HARDWARE_COLLECTIVE_COMMUNICATION_GROUP_H_
#define INFERRT_SRC_HARDWARE_COLLECTIVE_COMMUNICATION_GROUP_H_

#include <string>
#include <vector>
#include <memory>
#include "common/visible.h"

namespace mrt {
namespace collective {

class DA_API CommunicationGroup {
 public:
  explicit CommunicationGroup(const std::string &name, const std::vector<uint32_t> &groupRanks, uint32_t groupRank,
                              int64_t comm);

  ~CommunicationGroup() = default;

  virtual const std::string &group_name() const;
  virtual const std::vector<uint32_t> &group_ranks() const;
  virtual uint32_t group_rank() const;
  virtual uint32_t group_size() const;
  virtual int64_t communicator() const;

 protected:
  std::string groupName;
  std::vector<uint32_t> groupRanks;
  uint32_t groupRank;
  int64_t comm;
};

using CommunicationGroupPtr = std::shared_ptr<CommunicationGroup>;
}  // namespace collective
}  // namespace mrt

#endif  // INFERRT_SRC_HARDWARE_COLLECTIVE_COMMUNICATION_GROUP_H_
