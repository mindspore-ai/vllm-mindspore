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

#include <dirent.h>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <type_traits>

#include "common/logger.h"
#include "common/dynamic_lib_loader.h"
#include "ops/op_register.h"

namespace mrt {
namespace ops {
bool LoadOpLib(const std::string &opLibPrefix, std::stringstream *errMsg) {
  static std::unique_ptr<common::DynamicLibLoader> dynamicLibLoader = std::make_unique<common::DynamicLibLoader>();
  CHECK_IF_NULL(dynamicLibLoader);
  CHECK_IF_NULL(errMsg);
  DIR *dir = opendir(dynamicLibLoader->GetDynamicLibFilePath().c_str());
  if (dir == nullptr) {
    *errMsg << "Open Op Lib dir failed, file path:" << dynamicLibLoader->GetDynamicLibFilePath() << std::endl;
    return false;
  }
  struct dirent *entry;
  std::set<std::string> opLibs;
  while ((entry = readdir(dir)) != nullptr) {
    std::string opLibName = entry->d_name;
    if (opLibName.find(opLibPrefix) == std::string::npos) {
      continue;
    }
    if (opLibName.find_first_of(".") == std::string::npos) {
      continue;
    }
    opLibs.insert(opLibName);
  }
  for (const auto &opLibName : opLibs) {
    if (!dynamicLibLoader->LoadDynamicLib(opLibName, errMsg)) {
      return false;
    }
  }
  (void)closedir(dir);
  return true;
}

OpFactoryBase *OpFactoryBase::GetOpFactory(const std::string_view &name) {
  auto iter = OpFactoryMap().find(name);
  if (iter == OpFactoryMap().end()) {
    return nullptr;
  }
  return iter->second.get();
}

OpFactoryBase *OpFactoryBase::CreateOpFactory(const std::string_view &name, std::unique_ptr<OpFactoryBase> &&factory) {
  if (OpFactoryMap().find(name) != OpFactoryMap().end()) {
    LOG_EXCEPTION << name << " already has an OpFactory, please check!";
  }
  (void)OpFactoryMap().emplace(name, std::move(factory));
  return GetOpFactory(name);
}

OpFactoryBase::OpFactoryMapType &OpFactoryBase::OpFactoryMap() {
  // Functions containing static local variables should be implemented in .cc files to prevent multiple instances of the
  // same static variable in memory within a single process, which may occur when header files are included across
  // shared libraries.
  static OpFactoryBase::OpFactoryMapType factoryMap;
  return factoryMap;
}

}  // namespace ops
}  // namespace mrt
