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

static bool load_ascend_api = false;
static bool load_simulation_api = false;

void *GetLibHandler(const std::string &lib_path, bool if_global) {
  void *handler = nullptr;
  if (if_global) {
    handler = dlopen(lib_path.c_str(), RTLD_LAZY | RTLD_GLOBAL);
  } else {
    handler = dlopen(lib_path.c_str(), RTLD_LAZY | RTLD_LOCAL);
  }
  if (handler == nullptr) {
    LOG_OUT << "Dlopen " << lib_path << " failed!" << dlerror();
  }
  return handler;
}

std::string GetAscendPath() {
  Dl_info info;
  if (dladdr(reinterpret_cast<void *>(aclrtMalloc), &info) == 0) {
    LOG_ERROR << "Get dladdr failed.";
    return "";
  }
  auto path_tmp = std::string(info.dli_fname);
  const std::string kLatest = "latest";
  auto pos = path_tmp.rfind(kLatest);
  if (pos == std::string::npos) {
    LOG_ERROR << "Get ascend path failed, please check whether CANN packages are installed correctly, \n"
                 "and environment variables are set by source ${LOCAL_ASCEND}/ascend-toolkit/set_env.sh.";
  }
  return path_tmp.substr(0, pos) + kLatest + "/";
}

void LoadAscendApiSymbols() {
  if (load_ascend_api) {
    LOG_OUT << "Ascend api is already loaded.";
    return;
  }
  std::string ascend_path = GetAscendPath();
  LoadAclBaseApiSymbol(ascend_path);
  LoadAclOpCompilerApiSymbol(ascend_path);
  LoadAclMdlApiSymbol(ascend_path);
  LoadAclOpApiSymbol(ascend_path);
  LoadAclAllocatorApiSymbol(ascend_path);
  LoadAclRtApiSymbol(ascend_path);
  LoadAclApiSymbol(ascend_path);
  LoadAcltdtApiSymbol(ascend_path);
  load_ascend_api = true;
  LOG_OUT << "Load ascend api success!";
}

void LoadSimulationApiSymbols() {
  if (load_simulation_api) {
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
  load_simulation_api = true;
  LOG_OUT << "Load simulation api success!";
}
}  // namespace mrt::device::ascend
