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

#ifndef __OPS_UTILS_ASYNC_H__
#define __OPS_UTILS_ASYNC_H__

#include <string>
#include <functional>
#include "common/visible.h"

namespace mrt {
namespace ops {

using BindStreamFunc = std::function<void()>;
using ProcFunc = std::function<int()>;
using LaunchOpFunc =
  std::function<void(const std::string & /* op_name */, const ProcFunc & /* func */, bool /* sync */)>;
using WaitLaunchFinishFunc = std::function<void()>;

class DA_API OpAsync {
 public:
  static void SetLaunchOpFunc(const LaunchOpFunc &launchOpFunc);
  static const LaunchOpFunc &GetLaunchOpFunc();
  static void SetWaitLaunchFinishFunc(const WaitLaunchFinishFunc &waitLaunchFinishFunc);
  static WaitLaunchFinishFunc const &GetWaitLaunchFinishFunc();
};

}  // namespace ops
}  // namespace mrt

#endif  // __OPS_UTILS_ASYNC_H__