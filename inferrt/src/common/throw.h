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

#ifndef INFERRT_SRC_COMMON_THROW_H_
#define INFERRT_SRC_COMMON_THROW_H_

#include <sstream>

#include "common/logger.h"

namespace mrt {
namespace common {

// NOTE: We intentionally pass (file, line, func) from the call-site so the
// thrown exception points to the real error location rather than this helper.
template <typename... Args>
[[noreturn]] inline void ThrowExceptionAt(const char *file, int line, const char *func, const Args &...args) {
  std::ostringstream oss;
  (oss << ... << args);
  ::Cexception(file, line, func) << oss.str();
  __builtin_unreachable();
}

}  // namespace common
}  // namespace mrt

#define MRT_THROW(...) ::mrt::common::ThrowExceptionAt(__FILE__, __LINE__, __FUNCTION__, __VA_ARGS__)

#endif  // INFERRT_SRC_COMMON_THROW_H_
