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
#include "symbol_utils.h"
#include <string>
#include "acl_base_symbol.h"
#include "acl_compiler_symbol.h"
#include "acl_mdl_symbol.h"
#include "acl_op_symbol.h"
#include "acl_rt_allocator_symbol.h"
#include "acl_rt_symbol.h"
#include "acl_symbol.h"
#include "acl_tdt_symbol.h"

namespace mrt::device::ascend {

static bool loadAscendApi = false;
static bool loadSimulationApi = false;

void *GetLibHandler(const std::string &libPath, bool ifGlobal) {
  void *handler = nullptr;
  if (ifGlobal) {
    handler = dlopen(libPath.c_str(), RTLD_LAZY | RTLD_GLOBAL);
  } else {
    handler = dlopen(libPath.c_str(), RTLD_LAZY | RTLD_LOCAL);
  }
  if (handler == nullptr) {
    LOG_OUT << "Dlopen " << libPath << " failed!" << dlerror();
  }
  return handler;
}

std::string GetAscendPath() {
  Dl_info info;
  if (dladdr(reinterpret_cast<void *>(aclrtMalloc), &info) == 0) {
    LOG_ERROR << "Get dladdr failed.";
    return "";
  }
  auto pathTmp = std::string(info.dli_fname);
  const std::string kLatest = "latest";
  auto pos = pathTmp.rfind(kLatest);
  if (pos == std::string::npos) {
    LOG_ERROR << "Get ascend path failed, please check whether CANN packages are installed correctly, \n"
                 "and environment variables are set by source ${LOCAL_ASCEND}/ascend-toolkit/set_env.sh.";
  }
  return pathTmp.substr(0, pos) + kLatest + "/";
}

void LoadAscendApiSymbols() {
  if (loadAscendApi) {
    LOG_OUT << "Ascend api is already loaded.";
    return;
  }
  std::string ascendPath = GetAscendPath();
  LoadAclBaseApiSymbol(ascendPath);
  LoadAclOpCompilerApiSymbol(ascendPath);
  LoadAclMdlApiSymbol(ascendPath);
  LoadAclOpApiSymbol(ascendPath);
  LoadAclAllocatorApiSymbol(ascendPath);
  LoadAclRtApiSymbol(ascendPath);
  LoadAclApiSymbol(ascendPath);
  LoadAcltdtApiSymbol(ascendPath);
  loadAscendApi = true;
  LOG_OUT << "Load ascend api success!";
}

void LoadSimulationApiSymbols() {
  if (loadSimulationApi) {
    LOG_OUT << "Simulation api is already loaded.";
    return;
  }

  LoadSimulationAclBaseApi();
  LoadSimulationRtApi();
  LoadSimulationTdtApi();
  LoadSimulationAclOpCompilerApi();
  LoadSimulationAclMdlApi();
  LoadSimulationAclOpApi();
  LoadSimulationAclAllocatorApi();
  LoadSimulationAclApi();
  loadSimulationApi = true;
  LOG_OUT << "Load simulation api success!";
}
}  // namespace mrt::device::ascend
