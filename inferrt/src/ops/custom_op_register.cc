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

#include "ops/custom_op_register.h"
#include <mutex>
#include <algorithm>
#include <utility>
#include <vector>
#include <string>
#include <memory>

namespace mrt {
namespace ops {

CustomOpRegistry &CustomOpRegistry::GetInstance() {
  static CustomOpRegistry instance;
  return instance;
}

bool CustomOpRegistry::RegisterCustomOp(const std::string &op_name, CustomOpFactory &&factory) {
  if (custom_ops_.find(op_name) != custom_ops_.end()) {
    return false;
  }

  custom_ops_.emplace(op_name, std::move(factory));
  return true;
}

std::unique_ptr<Operator> CustomOpRegistry::CreateCustomOp(const std::string &op_name) {
  auto it = custom_ops_.find(op_name);
  if (it == custom_ops_.end()) {
    return nullptr;
  }

  return it->second();
}

bool CustomOpRegistry::IsCustomOpRegistered(const std::string &op_name) const {
  return custom_ops_.find(op_name) != custom_ops_.end();
}

std::vector<std::string> CustomOpRegistry::GetRegisteredOpNames() const {
  std::vector<std::string> names;
  names.reserve(custom_ops_.size());

  std::transform(custom_ops_.begin(), custom_ops_.end(), std::back_inserter(names),
                 [](const auto &pair) { return pair.first; });

  return names;
}

bool CustomOpRegistry::UnregisterCustomOp(const std::string &op_name) {
  auto it = custom_ops_.find(op_name);
  if (it != custom_ops_.end()) {
    custom_ops_.erase(it);
    return true;
  }

  return false;
}

std::unique_ptr<Operator> CreateCustomOperator(const std::string &name) {
  return CustomOpRegistry::GetInstance().CreateCustomOp(name);
}

}  // namespace ops
}  // namespace mrt
