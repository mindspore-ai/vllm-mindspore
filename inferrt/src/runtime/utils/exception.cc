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

#include "runtime/utils/exception.h"
#include "common/logger.h"

namespace mrt {
namespace runtime {
MrtException &MrtException::GetInstance() {
  static MrtException instance{};
  return instance;
}

void MrtException::SetException(const std::exception_ptr &exception) {
  std::lock_guard<std::mutex> lock(mtx_);
  if (exception_ != nullptr) {
    return;
  }

  if (exception != nullptr) {
    exception_ = exception;
  } else {
    exception_ = std::current_exception();
  }
}

void MrtException::CheckException() {
  std::lock_guard<std::mutex> lock(mtx_);
  if (exception_ != nullptr) {
    auto exception = exception_;
    exception_ = nullptr;
    std::rethrow_exception(exception);
  }
}
}  // namespace runtime
}  // namespace mrt
