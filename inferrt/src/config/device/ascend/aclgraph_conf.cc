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

#include "config/device/ascend/aclgraph_conf.h"
namespace mrt {
namespace config {
namespace ascend {

AclGraphConf &AclGraphConf::Instance() {
  static AclGraphConf instance;
  return instance;
}

void AclGraphConf::BeginCapture() { KernelCaptureExecutorManager::GetInstance().SetInCapture(true); }

void AclGraphConf::EndCapture() { KernelCaptureExecutorManager::GetInstance().SetInCapture(false); }

bool AclGraphConf::IsCapturing() const { return KernelCaptureExecutorManager::GetInstance().InCapture(); }

MempoolId_t AclGraphConf::GetPoolId() const { return KernelCaptureExecutorManager::GetInstance().PoolId(); }

void AclGraphConf::SetPoolId(MempoolId_t poolId) { KernelCaptureExecutorManager::GetInstance().SetPoolId(poolId); }

void AclGraphConf::SetOpCaptureSkip(const std::vector<std::string> &op_capture_skip) {
  KernelCaptureExecutorManager::GetInstance().SetOpCaptureSkip(op_capture_skip);
}
}  // namespace ascend
}  // namespace config
}  // namespace mrt
