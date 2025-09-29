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

#ifndef _WIN32
#include <dlfcn.h>
#else
#include <windows.h>
#endif
#include <limits.h>
#include <string>
#include <map>

#include "common/common.h"
#include "common/dynamic_lib_loader.h"

namespace mrt {
namespace common {
namespace {
std::string GetErrorMsg() {
#ifndef _WIN32
  const char *result = dlerror();
  return (result == nullptr) ? "Unknown" : result;
#else
  return std::to_string(GetLastError());
#endif
}
}  // namespace

DynamicLibLoader::~DynamicLibLoader() {
  for (const auto &[dlName, handle] : allHandles_) {
    if (dlclose(handle) != 0) {
      LOG_ERROR << "Closing dynamic lib: " << dlName << "failed, error message: " << GetErrorMsg();
    }
    LOG_OUT << "Close dynamic library: " << dlName << " successfully.";
  }
}

std::string DynamicLibLoader::GetFilePathFromDlInfo() {
  Dl_info dlInfo;
  if (dladdr(reinterpret_cast<void *>(DynamicLibLoader::GetFilePathFromDlInfo), &dlInfo) == 0) {
    LOG_ERROR << "Get file path by dladdr failed";
    return "";
  }
  std::string curSoPath = dlInfo.dli_fname;
  LOG_OUT << "Current so path : " << curSoPath;

  auto lastSlashPos = curSoPath.find_last_of("/");
  if (curSoPath.empty() || lastSlashPos == std::string::npos) {
    LOG_ERROR << "Current so path empty or the path [" << curSoPath << "] is invalid.";
    return "";
  }
  // During project build, place the current shared library (libmrt_common.so) and various plugins in
  // the same directory.
  auto dynamicLibPath = curSoPath.substr(0, lastSlashPos);
  LOG_OUT << "Current so dir path : " << dynamicLibPath;
  if (dynamicLibPath.size() >= PATH_MAX) {
    LOG_ERROR << "Current path [" << dynamicLibPath << "] is invalid.";
    return "";
  }
  char realPathMem[PATH_MAX] = {0};
  if (realpath(dynamicLibPath.c_str(), realPathMem) == nullptr) {
    LOG_ERROR << "Dynamic library path is invalid: [" << dynamicLibPath << "], skip!";
    return "";
  }
  return std::string(realPathMem);
}

bool DynamicLibLoader::LoadDynamicLib(const std::string &dlName, std::stringstream *errMsg) {
  CHECK_IF_NULL(errMsg);
  if (dlName.empty()) {
    LOG_ERROR << "Dynamic library name is empty";
    *errMsg << "Dynamic library name is empty" << std::endl;
    return false;
  }
  if (allHandles_.find(dlName) != allHandles_.end()) {
    LOG_OUT << "Dynamic library: " << dlName << " already loaded";
    return true;
  }
  void *handle = dlopen((filePath_ + "/" + dlName).c_str(), RTLD_LAZY | RTLD_LOCAL);
  if (handle == nullptr) {
    std::string errMsgStr = GetErrorMsg();
    LOG_ERROR << "Load dynamic library: " << dlName << " failed. " << errMsgStr;
    *errMsg << "Load dynamic library: " << dlName << " failed. " << errMsgStr << std::endl;
    return false;
  }
  allHandles_[dlName] = handle;
  LOG_OUT << "Load dynamic library: " << dlName << " successfully.";
  return true;
}

void DynamicLibLoader::CloseDynamicLib(const std::string &dlName) {
  if (allHandles_.find(dlName) == allHandles_.end()) {
    LOG_OUT << "Dynamic library: " << dlName << " not found";
    return;
  }
  if (dlclose(allHandles_[dlName]) != 0) {
    LOG_ERROR << "Closing dynamic lib: " << dlName << "failed, error message: " << GetErrorMsg();
  }
  allHandles_.erase(dlName);
  LOG_OUT << "Close dynamic library: " << dlName << " successfully.";
}

}  // namespace common
}  // namespace mrt
