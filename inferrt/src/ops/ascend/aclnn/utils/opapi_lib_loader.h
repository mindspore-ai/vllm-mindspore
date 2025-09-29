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

#ifndef __OPS_ASCEND_ACLNN_UTILS_OPAPI_LIB_LOADER_H__
#define __OPS_ASCEND_ACLNN_UTILS_OPAPI_LIB_LOADER_H__

#include <dlfcn.h>
#include <string>
#include <unordered_map>
#include <string_view>
#include <utility>
#include <tuple>
#include <type_traits>

#include "common/common.h"
#include "ops/ascend/aclnn/utils/aclnn_common_meta.h"

namespace mrt {
namespace ops {
inline constexpr const char *kNameOpApiLib = "lib64/libopapi.so";
extern std::unordered_map<void *, std::string> libHandlers;

void LoadOpApiLib();
void AclnnInit();
void AclnnFinalize();
void *GetAclnnOpApiFunc(const char *apiName);

inline void *GetOpApiLibHandler(const std::string &libPath) {
  auto handler = dlopen(libPath.c_str(), RTLD_LAZY);
  if (handler == nullptr) {
    LOG_OUT << "Dlopen " << libPath << " failed!" << dlerror();
  }
  return handler;
}

inline void *GetOpApiFuncFromLib(void *handler, const char *libName, const char *apiName) {
  CHECK_IF_NULL(handler);
  auto func = dlsym(handler, apiName);
  if (func == nullptr) {
    LOG_OUT << "Dlsym " << apiName << " from " << libName << " failed!" << dlerror();
  }
  return func;
}

template <typename T>
T LoadCommonMetaApi(const char *apiName) {
  for (auto &libHandler : libHandlers) {
    T apiFunc = reinterpret_cast<T>(GetOpApiFuncFromLib(libHandler.first, libHandler.second.c_str(), apiName));
    if (apiFunc == nullptr) {
      LOG_OUT << "Get CommonMetaApi [" << apiName << "] failed, libPath: " << libHandler.second;
    }
    return apiFunc;
  }
  return nullptr;
}

#define LOAD_COMMON_META_FUNC(name) name##_ = LoadCommonMetaApi<_##name##FuncPtr>(kName##name##_)

template <typename Tuple, size_t... I>
auto ConvertToOpApiFunc(const Tuple &params, void *opApiAddr, std::index_sequence<I...>) {
  using OpApiFunc = int (*)(typename std::decay<decltype(std::get<I>(params))>::type...);
  auto func = reinterpret_cast<OpApiFunc>(opApiAddr);
  return func;
}

template <typename Tuple>
auto ConvertToOpApiFunc(const Tuple &params, void *opApiAddr) {
  static constexpr auto size = std::tuple_size<Tuple>::value;
  return ConvertToOpApiFunc(params, opApiAddr, std::make_index_sequence<size>{});
}

template <typename Function, typename Tuple, size_t... I>
auto CallOpApiFunc(Function f, Tuple t, std::index_sequence<I...>) {
  return f(std::get<I>(t)...);
}

template <typename Function, typename Tuple>
auto CallOpApiFunc(Function f, Tuple t) {
  static constexpr auto size = std::tuple_size<Tuple>::value;
  return CallOpApiFunc(f, t, std::make_index_sequence<size>{});
}

}  // namespace ops
}  // namespace mrt
#endif  // __OPS_ASCEND_ACLNN_UTILS_OPAPI_LIB_LOADER_H__
