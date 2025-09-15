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

#include "hardware/hardware_abstract/device_context_manager.h"
#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#endif
#ifdef __linux__
#include <sys/wait.h>
#endif  // #ifdef __linux__
#include <dirent.h>
#include <algorithm>
#include <string>
#include <set>
#include <fstream>
#include "hardware/hardware_abstract/dlopen_macro.h"
#include "hardware/hardware_abstract/multi_stream_controller.h"
#include "common/logger.h"

namespace mrt {
namespace {
size_t constexpr GetStrLen(const char *const str) {
  if (*str == '\0') {
    return 0;
  } else {
    return GetStrLen(str + 1) + 1;
  }
}

constexpr auto kCudaHomeEnv = "CUDA_HOME";
constexpr auto kNvccVersionKeyWords = "Cuda compilation tools, release ";
constexpr size_t kNvccVersionKeyWordsSize = GetStrLen(kNvccVersionKeyWords);
constexpr auto kSuccessKeyWord = "Success";
constexpr size_t kSuccessKeyWordSize = GetStrLen(kSuccessKeyWord);
constexpr size_t kBufferSize = 999;
#if defined(_WIN32)
constexpr bool kIsWindowsPlatform = true;
#else
constexpr bool kIsWindowsPlatform = false;
#endif
}  // namespace
namespace device {
bool PluginLoader::LoadDynamicLib(const std::string &pluginFile, std::map<std::string, void *> *allHandles,
                                  std::stringstream *errMsg) {
  CHECK_IF_NULL(allHandles);
  CHECK_IF_NULL(errMsg);
  auto soName = GetDynamicLibName(pluginFile);
  void *handle = dlopen(pluginFile.c_str(), RTLD_LAZY | RTLD_LOCAL);
  if (handle == nullptr) {
    std::string errMsgStr = GetDlErrorMsg();
    LOG_OUT << "Load dynamic library: " << soName << " failed. " << errMsgStr;
    *errMsg << "Load dynamic library: " << soName << " failed. " << errMsgStr << std::endl;
    return false;
  }
  (*allHandles)[soName] = handle;
  return true;
}

void PluginLoader::CloseDynamicLib(const std::string &dlName, void *handle) {
  if (dlclose(handle) != 0) {
    LOG_ERROR << "Closing dynamic lib: " << dlName << "failed, error message: " << GetDlErrorMsg();
  }
}

bool PluginLoader::GetPluginPath(std::string *filePath) {
  CHECK_IF_NULL(filePath);
  Dl_info dlInfo;
  if (dladdr(reinterpret_cast<void *>(PluginLoader::GetPluginPath), &dlInfo) == 0) {
    LOG_ERROR << "Get file path by dladdr failed";
    return false;
  }
  std::string curSoPath = dlInfo.dli_fname;
  LOG_OUT << "Current so path : " << curSoPath;

  auto lastSlashPos = curSoPath.find_last_of("/");
  if (curSoPath.empty() || lastSlashPos == std::string::npos) {
    LOG_ERROR << "Current so path empty or the path [" << curSoPath << "] is invalid.";
    return false;
  }
  // During project build, place the current shared library (libhardware_abstract.so) and various hardware plugins in
  // the same directory.
  auto pluginSoPath = curSoPath.substr(0, lastSlashPos);
  LOG_OUT << "Current plugin so dir path : " << pluginSoPath;
  if (pluginSoPath.size() >= PATH_MAX) {
    LOG_ERROR << "Current path [" << pluginSoPath << "] is invalid.";
    return false;
  }
  char realPathMem[PATH_MAX] = {0};
  if (realpath(pluginSoPath.c_str(), realPathMem) == nullptr) {
    LOG_ERROR << "Plugin path is invalid: [" << pluginSoPath << "], skip!";
    return false;
  }
  *filePath = std::string(realPathMem);
  return true;
}

std::string PluginLoader::GetDynamicLibName(const std::string &pluginFile) {
  auto p1 = pluginFile.find_last_of("/") + 1;
  auto targetSo = pluginFile.substr(p1);
  return targetSo;
}

DeviceContextManager::~DeviceContextManager() { UnloadPlugin(); }

DeviceContextManager &DeviceContextManager::GetInstance() {
  static DeviceContextManager instance{};
  instance.LoadPlugin();
  return instance;
}

void DeviceContextManager::Register(const std::string &deviceName, DeviceContextCreator &&deviceContextCreator) {
  LOG_OUT << "Register device context creator for device: " << deviceName;
  if (deviceContextCreators_.find(deviceName) == deviceContextCreators_.end()) {
    (void)deviceContextCreators_.emplace(deviceName, deviceContextCreator);
  }
}

void DeviceContextManager::ClearDeviceContexts() {
  multiStreamControllers_.clear();
  for (auto &iter : deviceContexts_) {
    LOG_OUT << "Release device " << iter.first;
    if (iter.second == nullptr) {
      LOG_ERROR << "device context is null";
    }
    iter.second->Destroy();
  }
  backendToDeviceContext_.clear();
  deviceContexts_.clear();
}

void DeviceContextManager::ChildAfterFork() {
  LOG_OUT << "DeviceContextManager reinitialize after fork.";
  LOG_OUT << "Clear deviceContexts_.";
  deviceContexts_.clear();
  LOG_OUT << "DeviceContextManager reinitialize after fork done.";
}

void DeviceContextManager::BindDeviceCtx() const {
  for (auto &iter : deviceContexts_) {
    if (iter.second == nullptr) {
      LOG_ERROR << "device context is null";
    }
    if (iter.second->deviceResManager_ == nullptr) {
      LOG_ERROR << "device res manager is null";
    }
    if (!iter.second->deviceResManager_->BindDeviceToCurrentThread(true)) {
      LOG_ERROR << "Bind device failed";
    }
  }
}

DeviceContext *DeviceContextManager::GetOrCreateDeviceContext(const DeviceContextKey &deviceContextKey) {
  std::string deviceContextKeyStr = deviceContextKey.ToString();
  std::string name = deviceContextKey.deviceName_;

  auto deviceContextIter = deviceContexts_.find(deviceContextKeyStr);
  if (deviceContextIter != deviceContexts_.end()) {
    return deviceContextIter->second.get();
  }

  std::shared_ptr<DeviceContext> deviceContext;
  auto creatorIter = deviceContextCreators_.find(name);
  if (creatorIter != deviceContextCreators_.end()) {
    deviceContext = (creatorIter->second)(deviceContextKey);
    if (deviceContext == nullptr) {
      LOG_EXCEPTION << "Create device context failed, device context key: " << deviceContextKeyStr;
    }
    deviceContext->Initialize();
    if (deviceContext->deviceResManager_ == nullptr) {
      LOG_EXCEPTION << "Create device res manager failed, device context key: " << deviceContextKeyStr;
    }
    deviceContexts_[deviceContextKeyStr] = deviceContext;
    backendToDeviceContext_[name] = deviceContext;
    multiStreamControllers_[name] = std::make_shared<MultiStreamController>(deviceContext->deviceResManager_.get());
  } else {
    LOG_EXCEPTION << "Create device context failed, please make sure target device:" << name
                  << " is available, error message of loading plugins: " << GetErrorMsg();
  }
  return deviceContext.get();
}

DeviceContextPtr DeviceContextManager::GetDeviceContext(const std::string &deviceTarget) {
  if (backendToDeviceContext_.count(deviceTarget) == 0) {
    LOG_OUT << "Device context of device " << deviceTarget << " is not created yet.";
    return nullptr;
  }
  return backendToDeviceContext_[deviceTarget];
}

MultiStreamControllerPtr &DeviceContextManager::GetMultiStreamController(const std::string &deviceName) {
  auto &&iter = multiStreamControllers_.find(deviceName);
  if (iter != multiStreamControllers_.end()) {
    return iter->second;
  }
  LOG_ERROR << "Found multi stream controller failed, and try to initialize, deviceName : " << deviceName << ".";
  // use 0 temporarily.
  uint32_t deviceId = 0;
  DeviceContextKey hostKey = {deviceName, deviceId};
  const auto &realDeviceContext = GetOrCreateDeviceContext(hostKey);
  if (realDeviceContext == nullptr) {
    LOG_ERROR << "get or create device context failed";
  }
  auto &&iterAgain = multiStreamControllers_.find(deviceName);
  if (iterAgain == multiStreamControllers_.end()) {
    LOG_ERROR << "Get multi stream controller failed, deviceName : " << deviceName << ".";
  }
  return iterAgain->second;
}

void DeviceContextManager::WaitTaskFinishOnDevice() const {
  for (const auto &item : deviceContexts_) {
    auto deviceContext = item.second;
    try {
      if (deviceContext != nullptr && !deviceContext->deviceResManager_->SyncAllStreams()) {
        LOG_ERROR << "SyncStream failed";
        return;
      }
    } catch (const std::exception &ex) {
      LOG_ERROR << "SyncStream failed, exception:" << ex.what();
      return;
    }
  }
}

void DeviceContextManager::SyncAllStreams() const {
  for (const auto &item : deviceContexts_) {
    auto deviceContext = item.second;
    if (deviceContext != nullptr && !deviceContext->deviceResManager_->SyncAllStreams()) {
      LOG_ERROR << "SyncStream failed, device info: " << deviceContext->deviceContextKey().ToString();
    }
  }
}

std::string DeviceContextManager::GetErrorMsg() const { return dlopenErrorMsg_.str(); }

void DeviceContextManager::LoadPlugin() {
  if (loadInit_) {
    return;
  }
  loadInit_ = true;
  if (pluginPath_.empty() && !PluginLoader::GetPluginPath(&pluginPath_)) {
    LOG_ERROR << "Plugin path is invalid, skip!";
    dlopenErrorMsg_ << "Plugin path is invalid, skip!" << std::endl;
    return;
  }

  DIR *dir = opendir(pluginPath_.c_str());
  if (dir == nullptr) {
    LOG_ERROR << "Open plugin dir failed, plugin path:" << pluginPath_;
    dlopenErrorMsg_ << "Open plugin dir failed, plugin path:" << pluginPath_ << std::endl;
    return;
  }
  struct dirent *entry;
  std::map<std::string, std::set<std::string> > multiVersionPluginMap;  // key: plugin name, value: so file name
  while ((entry = readdir(dir)) != nullptr) {
    auto pluginFile = pluginPath_ + "/" + entry->d_name;
    constexpr auto pluginPrefix = "libhardware_";
    if (pluginFile.find(pluginPrefix) == std::string::npos) {
      continue;
    }
    if (pluginFile.find("libhardware_abstract") != std::string::npos) {
      continue;
    }
    std::string fileName = entry->d_name;
    auto dot = fileName.find_first_of(".");
    if (dot == std::string::npos) {
      continue;
    }
    (void)multiVersionPluginMap[fileName.substr(0, dot)].insert(pluginFile);
  }

  for (const auto &[pluginName, fileNames] : multiVersionPluginMap) {
    for (auto iter = fileNames.rbegin(); iter != fileNames.rend(); iter++) {
      const auto &fileName = *iter;
      auto ret = PluginLoader::LoadDynamicLib(fileName, &pluginMaps_, &dlopenErrorMsg_);
      if (ret) {
        LOG_OUT << "Load " << pluginName << " plugin file " << fileName << " successfully.";
      } else {
        LOG_ERROR << "Load " << pluginName << " plugin file " << fileName << " failed.";
      }
    }
  }
  (void)closedir(dir);
}

void DeviceContextManager::UnloadPlugin() {
  backendToDeviceContext_.clear();
  deviceContexts_.clear();
  deviceContextCreators_.clear();
  multiStreamControllers_.clear();

  if (pluginMaps_.empty()) {
    return;
  }
  auto iter = pluginMaps_.begin();
  while (iter != pluginMaps_.end()) {
    PluginLoader::CloseDynamicLib(iter->first, iter->second);
    (void)iter++;
  }
  pluginMaps_.clear();
}

}  // namespace device
}  // namespace mrt
