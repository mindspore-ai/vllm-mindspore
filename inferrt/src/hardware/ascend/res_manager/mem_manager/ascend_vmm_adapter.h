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

#ifndef INFERRT_SRC_HARDWARE_ASCEND_ASCEND_VMM_ADAPTER_H_
#define INFERRT_SRC_HARDWARE_ASCEND_ASCEND_VMM_ADAPTER_H_

#include <atomic>
#include <memory>
#include <map>
#include <vector>
#include <set>
#include <string>
#include <fstream>
#include <iostream>

#include "acl/acl.h"
#include "hardware/hardware_abstract/dlopen_macro.h"
#include "common/common.h"

#include "common/visible.h"

namespace mrt {
namespace device {
namespace ascend {
using DeviceMemPtr = void(*);
class MRT_EXPORT AscendVmmAdapter {
 public:
  static AscendVmmAdapter &GetInstance() {
    static AscendVmmAdapter instance{};
    return instance;
  }

  AscendVmmAdapter() {
    vmmAlignSize_ = kDefaultAlignSize;

    LOG_OUT << "VMM align size is " << vmmAlignSize_;
  }
  ~AscendVmmAdapter() = default;

 public:
  size_t GetRoundUpAlignSize(size_t input_size) const;
  size_t GetRoundDownAlignSize(size_t input_size) const;

  void ClearAllMemory();
  size_t AllocDeviceMem(size_t size, DeviceMemPtr *addr);
  size_t MmapDeviceMem(const size_t size, const DeviceMemPtr addr, const size_t max_size);
  size_t EagerFreeDeviceMem(const DeviceMemPtr addr, const size_t size);
  size_t GetAllocatedSize() { return physicalHandleSize_ * vmmAlignSize_; }

  size_t EmptyCache();

  static const bool IsEnabled() {
    static bool is_enable_vmm = IsVmmEnabled();
    return is_enable_vmm;
  }

 private:
  static const bool IsVmmEnabled() {
    if (!CheckVmmDriverVersion()) {
      return false;
    }

    LOG_OUT << "VMM is enabled.";
    return true;
  }

 private:
  uint64_t vmmAlignSize_;
  DeviceMemPtr FindVmmSegment(const DeviceMemPtr addr);
  size_t GetHandleSize(size_t input_size);
  std::atomic<size_t> physicalHandleSize_{0};
  std::map<DeviceMemPtr, aclrtDrvMemHandle> vmmMap_;
  std::vector<DeviceMemPtr> allReserveMems_;
  std::set<aclrtDrvMemHandle> cachedHandleSets_;
  static constexpr uint64_t kMB = 1024 * 1024;
  static constexpr uint64_t kDefaultAlignSize = 2 * kMB;
  static int StringToMB(const std::string &str) {
    std::stringstream ss(str);
    int num;
    std::string unit;
    if (!(ss >> num)) {
      LOG_ERROR << "No valid number could be extracted from the string, " << str;
    }
    if (!(ss >> unit) || unit != "MB") {
      LOG_ERROR << "The unit of the string is not MB, " << str;
    }
    if (ss.rdbuf()->in_avail() > 0) {
      LOG_ERROR << "The string has extra characters, " << str;
    }
    return num;
  }
  static bool CheckVmmDriverVersion() {
    // Get driver version
    constexpr auto ascend_install_info = "/etc/ascend_install.info";
    const std::string DRIVER_INSTALL_PATH_PARAM = "Driver_Install_Path_Param=";
    std::string driver_path = "/usr/local/Ascend";

    std::ifstream ascend_install_file(ascend_install_info);
    if (!ascend_install_file.is_open()) {
      LOG_OUT << "Open file " << ascend_install_info << " failed.";
    } else {
      std::string line;
      while (std::getline(ascend_install_file, line)) {
        size_t pos = line.find(DRIVER_INSTALL_PATH_PARAM);
        if (pos != std::string::npos) {
          // Extract the path after "Driver_Install_Path_Param="
          driver_path = line.substr(pos + DRIVER_INSTALL_PATH_PARAM.length());
          LOG_OUT << "Driver path is " << driver_path;
          break;
        }
      }
    }

    auto splitString = [](const std::string &str, char delimiter) -> std::vector<std::string> {
      std::vector<std::string> tokens;
      std::string token;
      std::istringstream tokenStream(str);
      while (std::getline(tokenStream, token, delimiter)) {
        tokens.push_back(token);
      }
      return tokens;
    };

    auto driver_version_info = driver_path + "/driver/version.info";
    const std::string DRIVER_VERSION_PARAM = "Version=";
    std::ifstream driver_version_file(driver_version_info);
    if (!driver_version_file.is_open()) {
      LOG_OUT << "Open file " << driver_version_info << " failed.";
    } else {
      std::string line;
      while (std::getline(driver_version_file, line)) {
        size_t pos = line.find(DRIVER_VERSION_PARAM);
        if (pos != std::string::npos) {
          // Extract the version after "Version="
          std::string driver_version = line.substr(pos + DRIVER_VERSION_PARAM.length());
          auto split_version = splitString(driver_version, '.');
          LOG_OUT << "Driver version is " << driver_version << ", major version is " << split_version[0];
          if (split_version[0] < "24") {
            LOG_OUT << "Driver version is less than 24.0.0, vmm is disabled by default, drvier_version: "
                    << driver_version;
            return false;
          }
          break;
        }
      }
    }
    return true;
  }
};
}  // namespace ascend
}  // namespace device
}  // namespace mrt

#endif
