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
#include "acl_compiler_symbol.h"
#include <string>
#include "symbol_utils.h"

namespace mrt::device::ascend {
aclopCompileAndExecuteFunObj aclopCompileAndExecute_ = nullptr;
aclopCompileAndExecuteV2FunObj aclopCompileAndExecuteV2_ = nullptr;
aclSetCompileoptFunObj aclSetCompileopt_ = nullptr;
aclopSetCompileFlagFunObj aclopSetCompileFlag_ = nullptr;
aclGenGraphAndDumpForOpFunObj aclGenGraphAndDumpForOp_ = nullptr;

void LoadAclOpCompilerApiSymbol(const std::string &ascendPath) {
  std::string complierPluginPath = ascendPath + "lib64/libacl_op_compiler.so";
  auto handler = GetLibHandler(complierPluginPath);
  if (handler == nullptr) {
    LOG_OUT << "Dlopen " << complierPluginPath << " failed!" << dlerror();
    return;
  }
  aclopCompileAndExecute_ = DlsymAscendFuncObj(aclopCompileAndExecute, handler);
  aclopCompileAndExecuteV2_ = DlsymAscendFuncObj(aclopCompileAndExecuteV2, handler);
  aclSetCompileopt_ = DlsymAscendFuncObj(aclSetCompileopt, handler);
  aclopSetCompileFlag_ = DlsymAscendFuncObj(aclopSetCompileFlag, handler);
  aclGenGraphAndDumpForOp_ = DlsymAscendFuncObj(aclGenGraphAndDumpForOp, handler);
  LOG_OUT << "Load acl op compiler api success!";
}

void LoadSimulationAclOpCompilerApi() {
  ASSIGN_SIMU(aclopCompileAndExecute);
  ASSIGN_SIMU(aclopCompileAndExecuteV2);
  ASSIGN_SIMU(aclSetCompileopt);
  ASSIGN_SIMU(aclopSetCompileFlag);
  ASSIGN_SIMU(aclGenGraphAndDumpForOp);
}
}  // namespace mrt::device::ascend
