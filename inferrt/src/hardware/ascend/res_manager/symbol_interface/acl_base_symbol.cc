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
#include "acl_base_symbol.h"
#include <string>
#include "symbol_utils.h"

namespace mrt::device::ascend {
aclCreateDataBufferFunObj aclCreateDataBuffer_ = nullptr;
aclCreateTensorDescFunObj aclCreateTensorDesc_ = nullptr;
aclDataTypeSizeFunObj aclDataTypeSize_ = nullptr;
aclDestroyDataBufferFunObj aclDestroyDataBuffer_ = nullptr;
aclDestroyTensorDescFunObj aclDestroyTensorDesc_ = nullptr;
aclGetTensorDescDimV2FunObj aclGetTensorDescDimV2_ = nullptr;
aclGetTensorDescNumDimsFunObj aclGetTensorDescNumDims_ = nullptr;
aclSetTensorConstFunObj aclSetTensorConst_ = nullptr;
aclSetTensorDescNameFunObj aclSetTensorDescName_ = nullptr;
aclSetTensorFormatFunObj aclSetTensorFormat_ = nullptr;
aclSetTensorPlaceMentFunObj aclSetTensorPlaceMent_ = nullptr;
aclSetTensorShapeFunObj aclSetTensorShape_ = nullptr;
aclrtGetSocNameFunObj aclrtGetSocName_ = nullptr;
aclUpdateDataBufferFunObj aclUpdateDataBuffer_ = nullptr;
aclGetDataBufferAddrFunObj aclGetDataBufferAddr_ = nullptr;
aclGetTensorDescSizeFunObj aclGetTensorDescSize_ = nullptr;
aclGetRecentErrMsgFunObj aclGetRecentErrMsg_ = nullptr;

void LoadAclBaseApiSymbol(const std::string &ascendPath) {
  std::string aclbasePluginPath = "lib64/libascendcl.so";
  auto baseHandler = GetLibHandler(ascendPath + aclbasePluginPath);
  if (baseHandler == nullptr) {
    LOG_OUT << "Dlopen " << aclbasePluginPath << " failed!" << dlerror();
    return;
  }
  aclCreateDataBuffer_ = DlsymAscendFuncObj(aclCreateDataBuffer, baseHandler);
  aclCreateTensorDesc_ = DlsymAscendFuncObj(aclCreateTensorDesc, baseHandler);
  aclDataTypeSize_ = DlsymAscendFuncObj(aclDataTypeSize, baseHandler);
  aclDestroyDataBuffer_ = DlsymAscendFuncObj(aclDestroyDataBuffer, baseHandler);
  aclDestroyTensorDesc_ = DlsymAscendFuncObj(aclDestroyTensorDesc, baseHandler);
  aclGetTensorDescDimV2_ = DlsymAscendFuncObj(aclGetTensorDescDimV2, baseHandler);
  aclGetTensorDescNumDims_ = DlsymAscendFuncObj(aclGetTensorDescNumDims, baseHandler);
  aclSetTensorConst_ = DlsymAscendFuncObj(aclSetTensorConst, baseHandler);
  aclSetTensorDescName_ = DlsymAscendFuncObj(aclSetTensorDescName, baseHandler);
  aclSetTensorFormat_ = DlsymAscendFuncObj(aclSetTensorFormat, baseHandler);
  aclSetTensorPlaceMent_ = DlsymAscendFuncObj(aclSetTensorPlaceMent, baseHandler);
  aclSetTensorShape_ = DlsymAscendFuncObj(aclSetTensorShape, baseHandler);
  aclrtGetSocName_ = DlsymAscendFuncObj(aclrtGetSocName, baseHandler);
  aclUpdateDataBuffer_ = DlsymAscendFuncObj(aclUpdateDataBuffer, baseHandler);
  aclGetDataBufferAddr_ = DlsymAscendFuncObj(aclGetDataBufferAddr, baseHandler);
  aclGetTensorDescSize_ = DlsymAscendFuncObj(aclGetTensorDescSize, baseHandler);
  aclGetRecentErrMsg_ = DlsymAscendFuncObj(aclGetRecentErrMsg, baseHandler);
  LOG_OUT << "Load acl base api success!";
}

void LoadSimulationAclBaseApi() {
  ASSIGN_SIMU(aclCreateDataBuffer);
  ASSIGN_SIMU(aclCreateTensorDesc);
  ASSIGN_SIMU(aclDataTypeSize);
  ASSIGN_SIMU(aclDestroyDataBuffer);
  ASSIGN_SIMU(aclDestroyTensorDesc);
  ASSIGN_SIMU(aclGetTensorDescDimV2);
  ASSIGN_SIMU(aclGetTensorDescNumDims);
  ASSIGN_SIMU(aclSetTensorConst);
  ASSIGN_SIMU(aclSetTensorDescName);
  ASSIGN_SIMU(aclSetTensorFormat);
  ASSIGN_SIMU(aclSetTensorPlaceMent);
  ASSIGN_SIMU(aclSetTensorShape);
  ASSIGN_SIMU(aclUpdateDataBuffer);
  ASSIGN_SIMU(aclrtGetSocName);
  ASSIGN_SIMU(aclGetDataBufferAddr);
  ASSIGN_SIMU(aclGetTensorDescSize);
  ASSIGN_SIMU(aclGetRecentErrMsg);
}
}  // namespace mrt::device::ascend
