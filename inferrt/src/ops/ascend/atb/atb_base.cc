/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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

#include "ops/ascend/atb/atb_base.h"
#include "hardware/hardware_abstract/device_context.h"
#include "hardware/hardware_abstract/collective/collective_manager.h"
#include "hardware/hardware_abstract/device_context_manager.h"

namespace mrt {
namespace ops {

AtbBase::~AtbBase() {
  for (auto &item : cache_) {
    if (item.second.op != nullptr) {
      atb::DestroyOperation(item.second.op);
    }
  }
}

OpsErrorCode AtbBase::GetWorkspaceSize(AtbCacheEntry &entry, atb::VariantPack &variant_pack, size_t *workspace_size) {
  CHECK_IF_NULL(entry.op);
  CHECK_IF_NULL(workspace_size);

  if (entry.workspace_cached) {
    *workspace_size = entry.workspace_size;
    return OpsErrorCode::SUCCESS;
  }

  auto deviceId = mrt::collective::CollectiveManager::Instance().local_rank_id();
  mrt::device::DeviceContextKey deviceContextKey = {"Ascend", deviceId};
  auto deviceContext = mrt::device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(deviceContextKey);
  CHECK_IF_NULL(deviceContext);
  CHECK_IF_NULL(deviceContext->deviceResManager_);
  auto stream = deviceContext->deviceResManager_->GetCurrentStream();
  CHECK_IF_NULL(stream);

  uint64_t ws_size = 0;
  auto ret = entry.op->Setup(variant_pack, ws_size, GetAtbContext(stream));
  if (ret != 0) {
    LOG_ERROR << "ATB Setup failed, ret: " << ret;
    return OpsErrorCode::UNKNOWN_ERROR;
  }

  entry.workspace_size = static_cast<size_t>(ws_size);
  entry.workspace_cached = true;
  *workspace_size = entry.workspace_size;
  return OpsErrorCode::SUCCESS;
}

OpsErrorCode AtbBase::LaunchAtb(atb::VariantPack variant_pack, void *workspace_ptr, size_t workspace_size,
                                aclrtStream stream) {
  CHECK_IF_NULL(op_);
  auto ret =
    op_->Execute(variant_pack, reinterpret_cast<uint8_t *>(workspace_ptr), workspace_size, GetAtbContext(stream));
  if (ret != 0) {
    LOG_ERROR << "ATB Execute failed, ret: " << ret;
    return OpsErrorCode::UNKNOWN_ERROR;
  }
  return OpsErrorCode::SUCCESS;
}

}  // namespace ops
}  // namespace mrt
