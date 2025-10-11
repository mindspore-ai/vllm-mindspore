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

#ifndef __OPS_ASCEND_ACLNN_UTILS_ACLNN_EXECUTOR_H__
#define __OPS_ASCEND_ACLNN_UTILS_ACLNN_EXECUTOR_H__

#include <string>
#include <utility>
#include <memory>
#include <functional>
#include <list>
#include <unordered_map>

#include "ops/ascend/aclnn/utils/aclnn_hash.h"
#include "ops/ascend/aclnn/utils/aclnn_cache.h"
#include "ops/ascend/aclnn/utils/opapi_lib_loader.h"
#include "ops/ascend/aclnn/utils/aclnn_converter.h"
#include "ops/ascend/aclnn/utils/aclnn_deleter.h"

namespace mrt {
namespace ops {
inline constexpr const char *kNameGetWorkspaceSize = "GetWorkspaceSize";
using RunOpApiFunc = int (*)(void *, uint64_t, aclOpExecutor *, const aclrtStream);

class AclnnExecutor {
 public:
  explicit AclnnExecutor(const std::string &&opApiName) : opApiName_(std::move(opApiName)) {
    getWorkspaceSizeApiName_ = opApiName_ + kNameGetWorkspaceSize;
    cacheEntryManager_ = ir::MakeIntrusive<CacheEntryManager>();
    AclnnInit();
  }
  ~AclnnExecutor() { AclnnFinalize(); }

  template <typename... Args>
  void GetWorkspaceSize(uint64_t *workspaceSize, const Args &...args) {
    auto hashId = CalcAclnnHash(opApiName_, args...);
    cacheEntry_ = cacheEntryManager_->GetCacheEntry(hashId);
    if (cacheEntry_ != nullptr) {
      LOG_OUT << opApiName_ << " hit cache with hashId: " << hashId;
      *workspaceSize = cacheEntry_->GetWorkspaceSize();
      return;
    }

    LOG_OUT << opApiName_ << " miss cache with hashId: " << hashId;
    auto [convertedParams, opExecutor] = GenerateOpExecutor(workspaceSize, args...);
    if (CheckExecutorRepeatable(opExecutor)) {
      cacheEntry_ = ir::MakeIntrusive<CacheEntryImpl<CacheProcessor<decltype(convertedParams)>>>(
        CacheProcessor<decltype(convertedParams)>(hashId, std::move(convertedParams), opExecutor, *workspaceSize));
      cacheEntryManager_->AddCacheEntry(hashId, cacheEntry_);
      LOG_OUT << opApiName_ << " cache the params with hashId: " << hashId;
      return;
    }
    ReleaseConvertedParams(convertedParams);
    ReleaseExecutor(opExecutor);
  }

  template <typename... Args>
  void Launch(void *workspace, size_t workspaceSize, void *stream, const Args &...args) {
    if (cacheEntry_ != nullptr) {
      CallUpdateAddr(cacheEntry_, args...);
      RunOpApi(workspace, workspaceSize, cacheEntry_->GetExecutor(), stream);
      return;
    }

    uint64_t workspaceSizeTmp = 0;
    auto [convertedParams, opExecutor] = GenerateOpExecutor(&workspaceSizeTmp, args...);
    RunOpApi(workspace, workspaceSize, opExecutor, stream);
    ReleaseConvertedParams(convertedParams);
    ReleaseExecutor(opExecutor);
  }

  template <typename... Args>
  auto GenerateOpExecutor(uint64_t *workspaceSize, const Args &...args) {
    const auto getWorkspaceSizeFuncPtr = GET_ACLNN_OP_FUNC(getWorkspaceSizeApiName_);
    if (getWorkspaceSizeFuncPtr == nullptr) {
      LOG_EXCEPTION << "Api " << getWorkspaceSizeApiName_ << " is not in " << kNameOpApiLib;
    }

    aclOpExecutor *opExecutor = nullptr;
    auto convertedParams = ConvertParams(args..., workspaceSize, &opExecutor);
    auto getWorkspaceSizeFunc = ConvertToOpApiFunc(convertedParams, getWorkspaceSizeFuncPtr);
    CHECK_IF_NULL(getWorkspaceSizeFunc);
    auto ret = CallOpApiFunc(getWorkspaceSizeFunc, convertedParams);
    if (ret != 0) {
      LOG_EXCEPTION << "Call " << getWorkspaceSizeApiName_ << " failed";
    }
    return std::make_tuple(convertedParams, opExecutor);
  }

  bool CheckExecutorRepeatable(aclOpExecutor *executor) {
    static const auto aclSetAclOpExecutorRepeatable = GET_ACLNN_COMMON_META_FUNC(aclSetAclOpExecutorRepeatable);
    if (aclSetAclOpExecutorRepeatable == nullptr) {
      LOG_OUT << "aclSetAclOpExecutorRepeatable is nullptr, which means the executor is not repeatable for op["
              << opApiName_ << "]";
      return false;
    }
    auto ret = aclSetAclOpExecutorRepeatable(executor);
    if (ret != 0) {
      LOG_OUT << "aclSetAclOpExecutorRepeatable failed, which means the executor is not repeatable for op["
              << opApiName_ << "]";
      return false;
    }
    LOG_OUT << "Set executor repeatable for op[" << opApiName_ << "] success";
    return true;
  }

  void RunOpApi(void *workspace, size_t workspaceSize, aclOpExecutor *opExecutor, void *stream) {
    const auto opApiFuncPtr = GET_ACLNN_OP_FUNC(opApiName_);
    if (opApiFuncPtr == nullptr) {
      LOG_EXCEPTION << "Api " << opApiName_ << " is not in " << kNameOpApiLib;
    }
    auto opApiFunc = reinterpret_cast<RunOpApiFunc>(opApiFuncPtr);
    auto opApiFuncRet = opApiFunc(workspace, workspaceSize, opExecutor, stream);
    if (opApiFuncRet != 0) {
      LOG_EXCEPTION << "Call " << opApiName_ << " failed";
    }
  }

 private:
  std::string opApiName_;
  std::string getWorkspaceSizeApiName_;
  CacheEntryPtr cacheEntry_{nullptr};
  CacheEntryManagerPtr cacheEntryManager_{nullptr};
};

}  // namespace ops
}  // namespace mrt
#endif  // __OPS_ASCEND_ACLNN_UTILS_ACLNN_EXECUTOR_H__
