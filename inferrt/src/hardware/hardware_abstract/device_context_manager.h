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

#ifndef INFERRT_SRC_HARDWARE_DEVICE_CONTEXT_MANAGER_H_
#define INFERRT_SRC_HARDWARE_DEVICE_CONTEXT_MANAGER_H_

#include <set>
#include <any>
#include <map>
#include <string>
#include <memory>
#include <utility>
#include <functional>
#include <mutex>
#include <vector>
#include "hardware/hardware_abstract/device_context.h"
#include "common/visible.h"

namespace mrt {
namespace device {
class MultiStreamController;
using DeviceContextCreator = std::function<std::shared_ptr<DeviceContext>(const DeviceContextKey &)>;
using MultiStreamControllerPtr = std::shared_ptr<MultiStreamController>;

class PluginLoader {
 public:
  static bool LoadDynamicLib(const std::string &pluginFile, std::map<std::string, void *> *allHandles,
                             std::stringstream *errMsg);
  static void CloseDynamicLib(const std::string &dlName, void *handle);
  static bool GetPluginPath(std::string *filePath);

 private:
  static std::string GetDynamicLibName(const std::string &pluginFile);
};

class MRT_EXPORT DeviceContextManager {
 public:
  static DeviceContextManager &GetInstance();
  ~DeviceContextManager();
  void Register(const std::string &deviceName, DeviceContextCreator &&deviceContextCreator);
  DeviceContext *GetOrCreateDeviceContext(const DeviceContextKey &deviceContextKey);
  // Return the device context of the specified device target.
  // The difference between this method and 'GetOrCreateDeviceContext' is this method only query device context by
  // device target(without device id) since inferrt only supports 'single process, single device'.
  DeviceContextPtr GetDeviceContext(const std::string &deviceTarget);
  MultiStreamControllerPtr &GetMultiStreamController(const std::string &deviceName);
  void ClearDeviceContexts();
  void ChildAfterFork();
  void WaitTaskFinishOnDevice() const;
  void SyncAllStreams() const;
  std::string GetErrorMsg() const;
  void BindDeviceCtx() const;

 private:
  DeviceContextManager() = default;
  void LoadPlugin();
  void UnloadPlugin();

  std::map<std::string, void *> pluginMaps_;
  bool loadInit_;
  std::string pluginPath_;

  // The string converted from DeviceContextKey -> DeviceContextPtr.
  std::map<std::string, DeviceContextPtr> deviceContexts_;
  // The name of device -> vector of DeviceContextPtr.
  std::map<std::string, DeviceContextPtr> backendToDeviceContext_;
  // The name of device -> DeviceContextCreator.
  std::map<std::string, DeviceContextCreator> deviceContextCreators_;
  // record error message of dlopen, print when create deviceContext failed.
  std::stringstream dlopenErrorMsg_;

  // Since multi device is not supported currently, here use device target type to improve performance.
  // Device target type : 0, 1, 2, 3, and real device support : 'Ascend' 'CPU'.
  std::map<std::string, MultiStreamControllerPtr> multiStreamControllers_;
};

class MRT_EXPORT DeviceContextRegister {
 public:
  DeviceContextRegister(const std::string &deviceName, DeviceContextCreator &&runtimeCreator) {
    DeviceContextManager::GetInstance().Register(deviceName, std::move(runtimeCreator));
  }
  ~DeviceContextRegister() = default;
};

#define MS_REGISTER_DEVICE(DEVICE_NAME, DEVICE_CONTEXT_CLASS)      \
  static const DeviceContextRegister g_device_##DEVICE_NAME##_reg( \
    DEVICE_NAME,                                                   \
    [](const DeviceContextKey &deviceContextKey) { return std::make_shared<DEVICE_CONTEXT_CLASS>(deviceContextKey); })
}  // namespace device
}  // namespace mrt
#endif  // INFERRT_SRC_HARDWARE_DEVICE_CONTEXT_MANAGER_H_
