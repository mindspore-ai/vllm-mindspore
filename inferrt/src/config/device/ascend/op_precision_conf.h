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

#ifndef __CONFIG_DEVICE_ASCEND_OP_PRECISION_CONF_H__
#define __CONFIG_DEVICE_ASCEND_OP_PRECISION_CONF_H__

#include <string>

#include "common/common.h"
#include "common/visible.h"
#include "config/device/ascend/common.h"

namespace mrt {
namespace config {
namespace ascend {
class MRT_EXPORT OpPrecisionConf {
 public:
  static OpPrecisionConf &Instance();

  void SetAclPrecisionMode(const std::string &aclPrecisionMode) { aclPrecisionMode_ = aclPrecisionMode; }

  const std::string &AclPrecisionMode() const { return aclPrecisionMode_; }

  void SetIsAllowMatmulHF32(bool isAllowMatmulHF32) { isAllowMatmulHF32_ = isAllowMatmulHF32; }

  bool IsAllowMatmulHF32() const { return isAllowMatmulHF32_; }

  void SetSocVersion(const SocVersion &socVersion) { socVersion_ = socVersion; }

  bool IsAllowFP32ToFP16();

 private:
  OpPrecisionConf() = default;
  DISABLE_COPY_AND_ASSIGN(OpPrecisionConf);

  std::string aclPrecisionMode_;
  bool isAllowMatmulHF32_;
  SocVersion socVersion_{SocVersion::UnsupportedSocVersion};
};

}  // namespace ascend
}  // namespace config
}  // namespace mrt
#endif  // __CONFIG_DEVICE_ASCEND_OP_PRECISION_CONF_H__
