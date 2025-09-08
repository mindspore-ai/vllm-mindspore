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
constexpr auto kGpuPluginName = "libinferrt_gpu";
#if defined(_WIN32)
constexpr bool kIsWindowsPlatform = true;
#else
constexpr bool kIsWindowsPlatform = false;
#endif
}  // namespace
namespace device {

DeviceContextManager &DeviceContextManager::GetInstance() {
  static DeviceContextManager instance{};
#ifdef WITH_BACKEND
  instance.LoadPlugin();
#endif
  return instance;
}

void DeviceContextManager::Register(const std::string &device_name, DeviceContextCreator &&device_context_creator) {
  LOG_OUT << "Register device context creator for device: " << device_name;
  if (deviceContextCreators_.find(device_name) == deviceContextCreators_.end()) {
    (void)deviceContextCreators_.emplace(device_name, device_context_creator);
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

DeviceContext *DeviceContextManager::GetOrCreateDeviceContext(const DeviceContextKey &device_context_key) {
  std::string device_context_key_str = device_context_key.ToString();
  std::string name = device_context_key.deviceName_;

  auto device_context_iter = deviceContexts_.find(device_context_key_str);
  if (device_context_iter != deviceContexts_.end()) {
    return device_context_iter->second.get();
  }

  std::shared_ptr<DeviceContext> device_context;
  auto creator_iter = deviceContextCreators_.find(name);
  if (creator_iter != deviceContextCreators_.end()) {
    device_context = (creator_iter->second)(device_context_key);
    if (device_context == nullptr) {
      LOG_ERROR << "create device context failed";
    }
    if (device_context->deviceResManager_ == nullptr) {
      LOG_ERROR << "create device res manager failed";
    }
    deviceContexts_[device_context_key_str] = device_context;
    backendToDeviceContext_[name] = device_context;
    multiStreamControllers_[name] =
      std::make_shared<MultiStreamController>(device_context->deviceResManager_.get());
  } else {
    LOG_ERROR << "Create device context failed, please make sure target device:" << name
              << " is available, error message of loading plugins: " << GetErrorMsg();
  }
  return device_context.get();
}

DeviceContextPtr DeviceContextManager::GetDeviceContext(const std::string &device_target) {
  if (backendToDeviceContext_.count(device_target) == 0) {
    LOG_OUT << "Device context of device " << device_target << " is not created yet.";
    return nullptr;
  }
  return backendToDeviceContext_[device_target];
}

MultiStreamControllerPtr &DeviceContextManager::GetMultiStreamController(const std::string &device_name) {
  auto &&iter = multiStreamControllers_.find(device_name);
  if (iter != multiStreamControllers_.end()) {
    return iter->second;
  }
  LOG_ERROR << "Found multi stream controller failed, and try to initialize, device_name : " << device_name << ".";
  // use 0 temporarily.
  uint32_t device_id = 0;
  DeviceContextKey host_key = {device_name, device_id};
  const auto &real_device_context = GetOrCreateDeviceContext(host_key);
  if (real_device_context == nullptr) {
    LOG_ERROR << "get or create device context failed";
  }
  auto &&iter_again = multiStreamControllers_.find(device_name);
  if (iter_again == multiStreamControllers_.end()) {
    LOG_ERROR << "Get multi stream controller failed, device_name : " << device_name << ".";
  }
  return iter_again->second;
}

void DeviceContextManager::WaitTaskFinishOnDevice() const {
  for (const auto &item : deviceContexts_) {
    auto device_context = item.second;
    try {
      if (device_context != nullptr && !device_context->deviceResManager_->SyncAllStreams()) {
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
    auto device_context = item.second;
    if (device_context != nullptr && !device_context->deviceResManager_->SyncAllStreams()) {
      LOG_ERROR << "SyncStream failed, device info: " << device_context->device_context_key().ToString();
    }
  }
}

std::string DeviceContextManager::GetErrorMsg() const { return dlopenErrorMsg_.str(); }
}  // namespace device
}  // namespace mrt
