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

#include "ops/ascend/aclnn/utils/aclnn_hash.h"
#include "ops/ascend/aclnn/utils/aclnn_cache.h"
#include "ops/ascend/aclnn/utils/opapi_lib_loader.h"
#include "ops/ascend/aclnn/utils/aclnn_converter.h"
#include "ops/ascend/aclnn/utils/aclnn_deleter.h"

namespace mrt {
namespace ops {
inline constexpr const char *kNameGetWorkspaceSize = "GetWorkspaceSize";
using RunOpFunc = int (*)(void *, uint64_t, aclOpExecutor *, const aclrtStream);

class AclnnExecutor {
 public:
  explicit AclnnExecutor(const std::string &&opApiName) : opApiName_(std::move(opApiName)) {
    getWorkspaceSizeApiName_ = opApiName_ + kNameGetWorkspaceSize;
    AclnnInit();
  }
  ~AclnnExecutor() { AclnnFinalize(); }

  template <typename OpClass, typename... Args>
  void GetWorkspaceSize(uint64_t *workspaceSize, const OpClass *op, const Args &...args) {
    // no cache
    if (cacheCapacity_ == 0) {
      GetWorkspaceSizeWithoutCacheList(workspaceSize, args...);
      return;
    }

    // TODO(linux): with cache list
    // hashId_ = CalcAclnnHash(opName_, args...);

    LOG_ERROR << "No cache list";
  }

  template <typename... Args>
  void GetWorkspaceSizeWithoutCacheList(uint64_t *workspaceSize, const Args &...args) {
    auto hashId = CalcAclnnHash(opApiName_, args...);
    if (isExecutorRepeatable_ && hashId == hashId_) {
      LOG_OUT << opApiName_ << " hit cache with hashId: " << hashId;
      *workspaceSize = workspaceSize_;
      return;
    }

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
    SetExecutorRepeatable(opExecutor);
    if (isExecutorRepeatable_) {
      workspaceSize_ = *workspaceSize;
      hashId_ = hashId;
      // cache the params
      cacheEntry_ = ir::MakeIntrusive<CacheEntryImpl<CacheProcessor<decltype(convertedParams)>>>(
        CacheProcessor<decltype(convertedParams)>(std::move(convertedParams), opExecutor));
      LOG_OUT << "Cache the params for op[" << opApiName_ << "] success";
      return;
    }
    hashId_ = 0;
    ReleaseConvertedParams(convertedParams);
    ReleaseExecutor(opExecutor);
    LOG_OUT << "Release the params and executor for op[" << opApiName_ << "] success";
  }

  template <typename... Args>
  void Launch(void *workspace, size_t workspaceSize, void *stream, const Args &...args) {
    if (isExecutorRepeatable_ && hashId_ != 0) {
      LaunchOpWithCache(workspace, workspaceSize, stream, args...);
    } else {
      LaunchOpWithoutCache(workspace, workspaceSize, stream, args...);
    }
  }

  template <typename... Args>
  void LaunchOpWithCache(void *workspace, size_t workspaceSize, void *stream, const Args &...args) {
    // update tensor addr
    CallUpdateAddr(cacheEntry_, args...);
    // run op
    const auto opApiFuncPtr = GET_ACLNN_OP_FUNC(opApiName_);
    if (opApiFuncPtr == nullptr) {
      LOG_EXCEPTION << "Api " << opApiName_ << " is not in " << kNameOpApiLib;
    }
    auto opApiFunc = reinterpret_cast<RunOpFunc>(opApiFuncPtr);
    auto opApiFuncRet = opApiFunc(workspace, workspaceSize, cacheEntry_->GetExecutor(), stream);
    if (opApiFuncRet != 0) {
      LOG_EXCEPTION << "Call " << opApiName_ << " failed";
    }
  }

  template <typename... Args>
  void LaunchOpWithoutCache(void *workspace, size_t workspaceSize, void *stream, const Args &...args) {
    // convert args and generate aclOpExecutor
    const auto getWorkspaceSizeFuncPtr = GET_ACLNN_OP_FUNC(getWorkspaceSizeApiName_);

    if (getWorkspaceSizeFuncPtr == nullptr) {
      LOG_EXCEPTION << "Api " << getWorkspaceSizeApiName_ << " is not in " << kNameOpApiLib;
    }
    uint64_t workspaceSizeTmp = 0;
    aclOpExecutor *opExecutor = nullptr;
    auto convertedParams = ConvertParams(args..., &workspaceSizeTmp, &opExecutor);
    auto getWorkspaceSizeFunc = ConvertToOpApiFunc(convertedParams, getWorkspaceSizeFuncPtr);
    CHECK_IF_NULL(getWorkspaceSizeFunc);
    auto getWorkspaceSizeRet = CallOpApiFunc(getWorkspaceSizeFunc, convertedParams);
    if (getWorkspaceSizeRet != 0) {
      LOG_EXCEPTION << "Call " << getWorkspaceSizeApiName_ << " failed";
    }

    // run op
    const auto opApiFuncPtr = GET_ACLNN_OP_FUNC(opApiName_);
    if (opApiFuncPtr == nullptr) {
      LOG_EXCEPTION << "Api " << opApiName_ << " is not in " << kNameOpApiLib;
    }
    auto opApiFunc = reinterpret_cast<RunOpFunc>(opApiFuncPtr);
    auto opApiFuncRet = opApiFunc(workspace, workspaceSize, opExecutor, stream);
    if (opApiFuncRet != 0) {
      LOG_EXCEPTION << "Call " << opApiName_ << " failed";
    }

    // release params and executor
    ReleaseConvertedParams(convertedParams);
    ReleaseExecutor(opExecutor);
  }

  void SetExecutorRepeatable(aclOpExecutor *executor) {
    static const auto aclSetAclOpExecutorRepeatable = GET_ACLNN_COMMON_META_FUNC(aclSetAclOpExecutorRepeatable);
    if (aclSetAclOpExecutorRepeatable == nullptr) {
      LOG_OUT << "aclSetAclOpExecutorRepeatable is nullptr, which means the executor is not repeatable for op["
              << opApiName_ << "]";
      isExecutorRepeatable_ = false;
      return;
    }
    auto ret = aclSetAclOpExecutorRepeatable(executor);
    if (ret != 0) {
      LOG_OUT << "aclSetAclOpExecutorRepeatable failed, which means the executor is not repeatable for op["
              << opApiName_ << "]";
      isExecutorRepeatable_ = false;
      return;
    }
    isExecutorRepeatable_ = true;
    LOG_OUT << "Set executor repeatable for op[" << opApiName_ << "] success";
  }

 private:
  std::string opApiName_;
  std::string getWorkspaceSizeApiName_;
  CacheEntryPtr cacheEntry_{nullptr};
  bool isExecutorRepeatable_{false};
  uint64_t workspaceSize_{0};
  uint64_t hashId_{0};
  size_t cacheCapacity_{0};
};

}  // namespace ops
}  // namespace mrt
#endif  // __OPS_ASCEND_ACLNN_UTILS_ACLNN_EXECUTOR_H__
