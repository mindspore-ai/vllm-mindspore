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
#include "acl_rt_allocator_symbol.h"
#include <string>
#include "symbol_utils.h"

namespace mrt::device::ascend {
aclrtAllocatorCreateDescFunObj aclrtAllocatorCreateDesc_ = nullptr;
aclrtAllocatorDestroyDescFunObj aclrtAllocatorDestroyDesc_ = nullptr;
aclrtAllocatorRegisterFunObj aclrtAllocatorRegister_ = nullptr;
aclrtAllocatorSetAllocAdviseFuncToDescFunObj aclrtAllocatorSetAllocAdviseFuncToDesc_ = nullptr;
aclrtAllocatorSetAllocFuncToDescFunObj aclrtAllocatorSetAllocFuncToDesc_ = nullptr;
aclrtAllocatorSetFreeFuncToDescFunObj aclrtAllocatorSetFreeFuncToDesc_ = nullptr;
aclrtAllocatorSetGetAddrFromBlockFuncToDescFunObj aclrtAllocatorSetGetAddrFromBlockFuncToDesc_ = nullptr;
aclrtAllocatorSetObjToDescFunObj aclrtAllocatorSetObjToDesc_ = nullptr;
aclrtAllocatorUnregisterFunObj aclrtAllocatorUnregister_ = nullptr;

void LoadAclAllocatorApiSymbol(const std::string &ascend_path) {
  std::string allocator_plugin_path = ascend_path + "lib64/libascendcl.so";
  auto handler = GetLibHandler(allocator_plugin_path);
  if (handler == nullptr) {
    LOG_OUT << "Dlopen " << allocator_plugin_path << " failed!" << dlerror();
    return;
  }
  aclrtAllocatorCreateDesc_ = DlsymAscendFuncObj(aclrtAllocatorCreateDesc, handler);
  aclrtAllocatorDestroyDesc_ = DlsymAscendFuncObj(aclrtAllocatorDestroyDesc, handler);
  aclrtAllocatorRegister_ = DlsymAscendFuncObj(aclrtAllocatorRegister, handler);
  aclrtAllocatorSetAllocAdviseFuncToDesc_ = DlsymAscendFuncObj(aclrtAllocatorSetAllocAdviseFuncToDesc, handler);
  aclrtAllocatorSetAllocFuncToDesc_ = DlsymAscendFuncObj(aclrtAllocatorSetAllocFuncToDesc, handler);
  aclrtAllocatorSetFreeFuncToDesc_ = DlsymAscendFuncObj(aclrtAllocatorSetFreeFuncToDesc, handler);
  aclrtAllocatorSetGetAddrFromBlockFuncToDesc_ =
    DlsymAscendFuncObj(aclrtAllocatorSetGetAddrFromBlockFuncToDesc, handler);
  aclrtAllocatorSetObjToDesc_ = DlsymAscendFuncObj(aclrtAllocatorSetObjToDesc, handler);
  aclrtAllocatorUnregister_ = DlsymAscendFuncObj(aclrtAllocatorUnregister, handler);
  LOG_OUT << "Load acl allocator api success!";
}

void LoadSimulationAclAllocatorApi() {
  ASSIGN_SIMU(aclrtAllocatorCreateDesc);
  ASSIGN_SIMU(aclrtAllocatorDestroyDesc);
  ASSIGN_SIMU(aclrtAllocatorRegister);
  ASSIGN_SIMU(aclrtAllocatorSetAllocAdviseFuncToDesc);
  ASSIGN_SIMU(aclrtAllocatorSetAllocFuncToDesc);
  ASSIGN_SIMU(aclrtAllocatorSetFreeFuncToDesc);
  ASSIGN_SIMU(aclrtAllocatorSetGetAddrFromBlockFuncToDesc);
  ASSIGN_SIMU(aclrtAllocatorSetObjToDesc);
  ASSIGN_SIMU(aclrtAllocatorUnregister);
}
}  // namespace mrt::device::ascend
