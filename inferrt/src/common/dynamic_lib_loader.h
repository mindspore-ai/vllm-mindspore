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

#ifndef __COMMON_DYNAMIC_LIB_LOADER_H__
#define __COMMON_DYNAMIC_LIB_LOADER_H__

#include <string>
#include <map>
#include <utility>
#include "common/common.h"
#include "common/visible.h"

namespace mrt {
namespace common {
class DA_API DynamicLibLoader {
 public:
  DynamicLibLoader() {
    filePath_ = GetFilePathFromDlInfo();
    if (filePath_.empty()) {
      LOG_ERROR << "Get dynamic library file path by dladdr failed";
    }
  }
  explicit DynamicLibLoader(const std::string &&filePath) : filePath_(std::move(filePath)) {}
  ~DynamicLibLoader();

  bool LoadDynamicLib(const std::string &dlName, std::stringstream *errMsg);
  void CloseDynamicLib(const std::string &dlName);

  const std::string &GetDynamicLibFilePath() const { return filePath_; }

 private:
  DISABLE_COPY_AND_ASSIGN(DynamicLibLoader)
  static std::string GetFilePathFromDlInfo();
  std::map<std::string, void *> allHandles_;
  std::string filePath_;
};
}  // namespace common
}  // namespace mrt

#endif  // __COMMON_DYNAMIC_LIB_LOADER_H__
