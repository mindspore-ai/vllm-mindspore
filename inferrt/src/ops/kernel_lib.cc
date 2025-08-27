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

#include <utility>
#include <string>
#include <memory>
#include <functional>
#include <unordered_map>
#ifndef _WIN32
#include <libgen.h>
#include <dlfcn.h>
#else
#include <windows.h>
#endif

#include "common/common.h"
#include "ops/kernel_lib.h"

namespace mrt {
namespace ops {

KernelLibRegistry &KernelLibRegistry::Instance() {
  static KernelLibRegistry instance{};
  return instance;
}

KernelLibRegistry::~KernelLibRegistry() {
  for (auto &iter : kernelLibs_) {
    delete iter.second;
  }
  kernelLibs_.clear();
}

void KernelLibRegistry::Register(const std::string &name, const KernelLibCreator &&creator) {
  if (kernelLibCreators_.find(name) == kernelLibCreators_.end()) {
    LOG_OUT << "KernelLibCreator for " << name << " registered.";
    (void)kernelLibCreators_.emplace(name, std::move(creator));
  }
}

void KernelLibRegistry::Load(const std::string &path) {
  if (kernelLibHandles_.count(path) > 0) {
    return;
  }

  LOG_OUT << "Load kernel lib path: " << path;
  void *handle;
  std::string errMsg = "";
#ifndef _WIN32
  handle = dlopen(path.c_str(), RTLD_LAZY);
  const char *result = dlerror();
  errMsg = (result == nullptr) ? "Unknown" : result;
#else
  handle = LoadLibrary(path.c_str());
  errMsg = std::to_string(GetLastError());
#endif

  if (handle == nullptr) {
    LOG_ERROR << "Load " + path + " failed, error: " + errMsg;
    return;
  }
  (void)kernelLibHandles_.emplace(path, handle);
}

const KernelLib *KernelLibRegistry::Get(const std::string &name) {
  if (auto iter = kernelLibs_.find(name); iter != kernelLibs_.end()) {
    return iter->second;
  }
  if (auto iter = kernelLibCreators_.find(name); iter != kernelLibCreators_.end()) {
    auto kernelLib = (iter->second)();
    kernelLibs_[name] = kernelLib;
    return kernelLib;
  }
  LOG_ERROR << "kernelLibCreators_: " << &kernelLibCreators_;
  for (auto &pair : kernelLibCreators_) {
    LOG_ERROR << "kernelLibCreators_: " << pair.first;
  }
  LOG_ERROR << "KernelLib " << name << " is not exist.";
  return nullptr;
}

}  // namespace ops
}  // namespace mrt
