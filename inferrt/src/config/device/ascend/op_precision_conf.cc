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

#include "config/device/ascend/op_precision_conf.h"

namespace mrt {
namespace config {
namespace ascend {
constexpr auto kAclMustKeepOriginDtype = "must_keep_origin_dtype";
constexpr auto kAclAllowFP32ToFP16 = "allow_fp32_to_fp16";

OpPrecisionConf &OpPrecisionConf::Instance() {
  static OpPrecisionConf instance;
  return instance;
}

bool OpPrecisionConf::IsAllowFP32ToFP16() {
  bool ret = socVersion_ < SocVersion::k910B1;
  if (!aclPrecisionMode_.empty()) {
    if (aclPrecisionMode_ == kAclMustKeepOriginDtype) {
      ret = false;
    } else if (aclPrecisionMode_ == kAclAllowFP32ToFP16) {
      ret = true;
    } else {
      LOG_OUT << "Unsupported precision mode: " << aclPrecisionMode_;
    }
  }
  return ret;
}

}  // namespace ascend
}  // namespace config
}  // namespace mrt
