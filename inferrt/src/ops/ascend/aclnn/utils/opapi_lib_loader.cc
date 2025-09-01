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

#include <string>
#include <mutex>
#include <shared_mutex>
#include <vector>
#include <unordered_map>

#include "ops/ascend/aclnn/utils/opapi_lib_loader.h"
#include "hardware/ascend/res_manager/symbol_interface/symbol_utils.h"

namespace mrt {
namespace ops {
namespace {
void LoadCommonMetaFuncApi() {
  LOAD_COMMON_META_FUNC(aclCreateTensor);
  LOAD_COMMON_META_FUNC(aclCreateScalar);
  LOAD_COMMON_META_FUNC(aclCreateIntArray);
  LOAD_COMMON_META_FUNC(aclCreateFloatArray);
  LOAD_COMMON_META_FUNC(aclCreateBoolArray);
  LOAD_COMMON_META_FUNC(aclCreateTensorList);

  LOAD_COMMON_META_FUNC(aclDestroyTensor);
  LOAD_COMMON_META_FUNC(aclDestroyScalar);
  LOAD_COMMON_META_FUNC(aclDestroyIntArray);
  LOAD_COMMON_META_FUNC(aclDestroyFloatArray);
  LOAD_COMMON_META_FUNC(aclDestroyBoolArray);
  LOAD_COMMON_META_FUNC(aclDestroyTensorList);
  LOAD_COMMON_META_FUNC(aclDestroyAclOpExecutor);

  LOAD_COMMON_META_FUNC(aclnnInit);
  LOAD_COMMON_META_FUNC(aclnnFinalize);

  LOAD_COMMON_META_FUNC(aclSetAclOpExecutorRepeatable);

  LOAD_COMMON_META_FUNC(aclSetTensorAddr);
  LOAD_COMMON_META_FUNC(aclSetDynamicTensorAddr);
}
}  // namespace

static bool isLoaded = false;
static bool isAclnnInit = false;
static std::mutex initMutex;
static std::shared_mutex rwOpApiMutex;
// handler -> libPath
std::unordered_map<void *, std::string> libHandlers;

void LoadOpApiLib() {
  if (isLoaded) {
    return;
  }
  const std::string ascendPath = device::ascend::GetAscendPath();
  const std::vector<std::string> dependLibs = {"libdummy_tls.so", "libnnopbase.so"};
  std::unique_lock<std::shared_mutex> writeLock(rwOpApiMutex);
  for (const auto &depLib : dependLibs) {
    (void)GetOpApiLibHandler(ascendPath + "lib64/" + depLib);
  }
  auto opApiLibPath = ascendPath + kNameOpApiLib;
  auto handler = GetOpApiLibHandler(opApiLibPath);
  if (handler != nullptr) {
    LOG_OUT << "Load lib " << opApiLibPath << " success";
    (void)libHandlers.emplace(handler, opApiLibPath);
  }
  LoadCommonMetaFuncApi();
  isLoaded = true;
  LOG_OUT << "Load opapi lib success";
}

void *GetAclnnOpApiFunc(const char *apiName) {
  // apiName -> api
  static thread_local std::unordered_map<std::string, void *> opapiCache;
  auto iter = opapiCache.find(std::string(apiName));
  if (iter != opapiCache.end()) {
    LOG_OUT << "OpApi " << apiName << " hit cache";
    return iter->second;
  }
  std::shared_lock<std::shared_mutex> readLock(rwOpApiMutex);
  if (libHandlers.size() == 0) {
    readLock.unlock();
    LoadOpApiLib();
  }
  for (auto &libHandler : libHandlers) {
    auto apiFunc = GetOpApiFuncFromLib(libHandler.first, libHandler.second.c_str(), apiName);
    if (apiFunc != nullptr) {
      (void)opapiCache.emplace(std::string(apiName), apiFunc);
      LOG_OUT << "Get OpApiFunc [" << apiName << "] from " << libHandler.second;
      return apiFunc;
    }
  }
  LOG_OUT << "Dlsym " << apiName << " failed";
  (void)opapiCache.emplace(std::string(apiName), nullptr);
  return nullptr;
}

void AclnnInit() {
  std::lock_guard<std::mutex> lock(initMutex);
  if (isAclnnInit) {
    return;
  }
  static const auto aclnnInit = GET_ACLNN_COMMON_META_FUNC(aclnnInit);
  CHECK_IF_NULL(aclnnInit);
  auto ret = aclnnInit(nullptr);
  CHECK_IF_FAIL(ret == 0);
  isAclnnInit = true;
  LOG_OUT << "Aclnn init success";
}

void AclnnFinalize() {
  if (!isAclnnInit) {
    return;
  }
  static const auto aclnnFinalize = GET_ACLNN_COMMON_META_FUNC(aclnnFinalize);
  CHECK_IF_NULL(aclnnFinalize);
  auto ret = aclnnFinalize();
  CHECK_IF_FAIL(ret == 0);
  isAclnnInit = false;
  LOG_OUT << "Aclnn finalize success";
}

}  // namespace ops
}  // namespace mrt
