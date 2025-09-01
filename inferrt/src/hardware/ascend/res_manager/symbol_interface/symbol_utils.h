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

#ifndef INFERRT_SRC_HARDWARE_ASCEND_SYMBOL_UTILS_H_
#define INFERRT_SRC_HARDWARE_ASCEND_SYMBOL_UTILS_H_
#include <string>
#include "common/common.h"
#include "acl/acl.h"
#include "common/visible.h"

#ifndef ACL_ERROR_RT_DEVICE_MEM_ERROR
#define ACL_ERROR_RT_DEVICE_MEM_ERROR 507053
#endif
#ifndef ACL_ERROR_RT_HBM_MULTI_BIT_ECC_ERROR
#define ACL_ERROR_RT_HBM_MULTI_BIT_ECC_ERROR 507054
#endif
#ifndef ACL_ERROR_RT_COMM_OP_RETRY_FAIL
#define ACL_ERROR_RT_COMM_OP_RETRY_FAIL 507904
#endif
#ifndef ACL_ERROR_RT_DEVICE_TASK_ABORT
#define ACL_ERROR_RT_DEVICE_TASK_ABORT 107022
#endif
const int thread_level = 0;

template <typename Function, typename... Args>
auto RunAscendApi(Function f, int line, const char *callF, const char *funcName, Args... args) {
  if (f == nullptr) {
    LOG_ERROR << funcName << " is null.";
  }

  if constexpr (std::is_same_v<std::invoke_result_t<decltype(f), Args...>, int>) {
    auto ret = f(args...);
    return ret;
  } else {
    return f(args...);
  }
}

template <typename Function>
auto RunAscendApi(Function f, int line, const char *callF, const char *funcName) {
  if (f == nullptr) {
    LOG_ERROR << funcName << " is null.";
  }
  if constexpr (std::is_same_v<std::invoke_result_t<decltype(f)>, int>) {
    auto ret = f();
    return ret;
  } else {
    return f();
  }
}

template <typename Function>
bool HasAscendApi(Function f) {
  return f != nullptr;
}

namespace mrt::device::ascend {

#define CALL_ASCEND_API(funcName, ...) \
  RunAscendApi(mrt::device::ascend::funcName##_, __LINE__, __FUNCTION__, #funcName, ##__VA_ARGS__)

#define HAS_ASCEND_API(funcName) HasAscendApi(mrt::device::ascend::funcName##_)

DA_API std::string GetAscendPath();
void *GetLibHandler(const std::string &libPath, bool ifGlobal = false);
void LoadAscendApiSymbols();
void LoadSimulationApiSymbols();
}  // namespace mrt::device::ascend

#endif  // INFERRT_SRC_HARDWARE_ASCEND_SYMBOL_UTILS_H_
