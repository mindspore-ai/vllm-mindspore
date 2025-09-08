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
#include "acl_symbol.h"
#include <string>
#include "symbol_utils.h"

namespace mrt::device::ascend {

aclInitFunObj aclInit_ = nullptr;
aclFinalizeFunObj aclFinalize_ = nullptr;

void LoadAclApiSymbol(const std::string &ascendPath) {
  std::string aclPluginPath = ascendPath + "lib64/libascendcl.so";
  auto baseHandler = GetLibHandler(aclPluginPath);
  if (baseHandler == nullptr) {
    LOG_OUT << "Dlopen " << aclPluginPath << " failed!" << dlerror();
    return;
  }
  aclInit_ = DlsymAscendFuncObj(aclInit, baseHandler);
  aclFinalize_ = DlsymAscendFuncObj(aclFinalize, baseHandler);
  LOG_OUT << "Load acl base api success!";
}

void LoadSimulationAclApi() {
  ASSIGN_SIMU(aclInit);
  ASSIGN_SIMU(aclFinalize);
}
}  // namespace mrt::device::ascend
