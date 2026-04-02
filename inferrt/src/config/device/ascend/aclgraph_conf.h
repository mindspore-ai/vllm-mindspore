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

#ifndef __CONFIG_DEVICE_ASCEND_ACLGRAPH_CONF_H__
#define __CONFIG_DEVICE_ASCEND_ACLGRAPH_CONF_H__

#include <string>

#include "common/common.h"
#include "common/visible.h"
#include "config/device/ascend/common.h"
#include "runtime/executor/kernel_capture/kernel_capture_executor.h"

namespace mrt {
namespace config {
namespace ascend {
using MempoolId_t = mrt::runtime::MempoolId_t;
using KernelCaptureExecutorManager = mrt::runtime::KernelCaptureExecutorManager;
class MRT_EXPORT AclGraphConf {
 public:
  static AclGraphConf &Instance();

  void BeginCapture();

  void EndCapture();

  bool IsCapturing() const;

  MempoolId_t GetPoolId() const;

  void SetPoolId(MempoolId_t poolId);

  void SetOpCaptureSkip(const std::vector<std::string> &op_capture_skip = {});

 private:
  AclGraphConf() = default;
  DISABLE_COPY_AND_ASSIGN(AclGraphConf);
};

}  // namespace ascend
}  // namespace config
}  // namespace mrt
#endif  // __CONFIG_DEVICE_ASCEND_ACLGRAPH_CONF_H__
