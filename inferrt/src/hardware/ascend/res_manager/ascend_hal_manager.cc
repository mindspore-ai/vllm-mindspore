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

#include "hardware/ascend/res_manager/ascend_hal_manager.h"

#include <unistd.h>
#include <fstream>
#include <string>
#include "common/common.h"
#include "acl/acl_rt.h"

#include "hardware/ascend/res_manager/symbol_interface/acl_rt_symbol.h"
#include "hardware/ascend/res_manager/symbol_interface/acl_symbol.h"
#include "hardware/ascend/res_manager/symbol_interface/symbol_utils.h"

namespace mrt {
namespace device {
namespace ascend {
namespace {
constexpr auto kSaturationMode = "Saturation";
constexpr auto kINFNANMode = "INFNAN";
}  // namespace
static thread_local aclrtContext thread_local_rt_context{nullptr};

AscendHalManager AscendHalManager::instance_{};
AscendHalManager &AscendHalManager::GetInstance() { return instance_; }

void AscendHalManager::InitDevice(uint32_t device_id) {
  LOG_OUT << "Enter SetRtDevice, current initialize device number:" << initializedDeviceSet_.size();
  if (initializedDeviceSet_.find(device_id) != initializedDeviceSet_.end()) {
    LOG_OUT << "Device " << device_id << " has been set";
    return;
  }

  auto ret = CALL_ASCEND_API(aclrtSetDevice, UintToInt(device_id));
  if (ret != ACL_SUCCESS) {
    auto device_count = GetDeviceCount();
    LOG_ERROR << "Call aclrtSetDevice failed, ret[" << static_cast<int>(ret) << "]. Got device count[" << device_count
              << "] and device id[" << device_id << "], please check if device id is valid.";
  }

  aclrtContext rt_context;
  ret = CALL_ASCEND_API(aclrtGetCurrentContext, &rt_context);
  if (ret != ACL_SUCCESS || rt_context == nullptr) {
    LOG_ERROR << "Call aclrtGetCurrentContext failed, ret[" << ret << "]";
    return;
  }

  defaultDeviceContextMap_[device_id] = rt_context;
  (void)initializedDeviceSet_.insert(device_id);
}

void AscendHalManager::ResetDevice(uint32_t device_id) {
  if (initializedDeviceSet_.find(device_id) != initializedDeviceSet_.end()) {
    auto ret = CALL_ASCEND_API(aclrtResetDevice, UintToInt(device_id));
    if (ret != ACL_SUCCESS) {
      LOG_ERROR << "Call aclrtResetDevice, ret[" << ret << "]";
    }
    defaultDeviceContextMap_[device_id] = nullptr;
    (void)initializedDeviceSet_.erase(device_id);
  }
}

uint32_t AscendHalManager::GetDeviceCount() {
  uint32_t device_count = 0;
  auto ret = CALL_ASCEND_API(aclrtGetDeviceCount, &device_count);
  if (ret != ACL_SUCCESS) {
    LOG_ERROR << "Call rtGetDeviceCount, ret[" << static_cast<int>(ret) << "]";
  }
  return device_count;
}

void AscendHalManager::SetDeviceSatMode(const aclrtFloatOverflowMode &overflow_mode) {
  auto overflow_mode_str =
    (overflow_mode == aclrtFloatOverflowMode::ACL_RT_OVERFLOW_MODE_SATURATION) ? kSaturationMode : kINFNANMode;
  LOG_OUT << "The current overflow detection mode is " << overflow_mode_str << ".";
  auto ret = CALL_ASCEND_API(aclrtSetDeviceSatMode, overflow_mode);
  if (ret != ACL_SUCCESS) {
    LOG_ERROR << "Set " << overflow_mode_str << " mode failed.";
  }
}

void AscendHalManager::SetOpWaitTimeout(uint32_t op_wait_timeout) {
  LOG_OUT << "Set op wait timeout: " << op_wait_timeout << " s";
  auto acl_ret = CALL_ASCEND_API(aclrtSetOpWaitTimeout, op_wait_timeout);
  if (acl_ret != ACL_SUCCESS) {
    LOG_ERROR << "Set op wait timeout failed, error: " << acl_ret;
  }
}

void AscendHalManager::SetOpExecuteTimeOut(uint32_t op_execute_timeout) {
  LOG_OUT << "Set op execute timeout: " << op_execute_timeout << " s";
  auto acl_ret = CALL_ASCEND_API(aclrtSetOpExecuteTimeOut, op_execute_timeout);
  if (acl_ret != ACL_SUCCESS) {
    LOG_ERROR << "Set op execute timeout failed, error: " << acl_ret;
  }
}

aclrtContext AscendHalManager::CreateContext(uint32_t device_id) {
  aclrtContext rt_context;
  auto ret = CALL_ASCEND_API(aclrtCreateContext, &rt_context, device_id);
  if (ret != ACL_SUCCESS) {
    LOG_ERROR << "Call aclrtCreateContext failed, ret: " << ret;
  }
  rtContexts_.insert(rt_context);
  return rt_context;
}

void AscendHalManager::ResetContext(uint32_t device_id) {
  aclrtContext rt_context = CreateContext(device_id);
  defaultDeviceContextMap_[device_id] = rt_context;
}

void AscendHalManager::DestroyContext(aclrtContext context) {
  auto ret = CALL_ASCEND_API(aclrtDestroyContext, context);
  if (ret != ACL_SUCCESS) {
    LOG_ERROR << "Failed to destroy context, ret = " << ret << ".";
  }
  rtContexts_.erase(context);
}

void AscendHalManager::DestroyAllContext() {
  for (auto context : rtContexts_) {
    auto ret = CALL_ASCEND_API(aclrtDestroyContext, context);
    if (ret != ACL_SUCCESS) {
      LOG_ERROR << "Failed to destroy context, ret = " << ret << ".";
    }
  }
  rtContexts_.clear();
}

void AscendHalManager::SetContextForce(uint32_t device_id) {
  if (defaultDeviceContextMap_[device_id] == nullptr) {
    return;
  }
  auto ret = CALL_ASCEND_API(aclrtSetCurrentContext, defaultDeviceContextMap_[device_id]);
  if (ret != ACL_SUCCESS) {
    LOG_ERROR << "Call aclrtSetCurrentContext, ret[" << ret << "]";
  }
}

void AscendHalManager::SetContext(uint32_t device_id) {
  if (defaultDeviceContextMap_[device_id] == nullptr) {
    return;
  }
  if (thread_local_rt_context == defaultDeviceContextMap_[device_id]) {
    return;
  }
  auto ret = CALL_ASCEND_API(aclrtSetCurrentContext, defaultDeviceContextMap_[device_id]);
  if (ret != ACL_SUCCESS) {
    LOG_ERROR << "Call aclrtSetCurrentContext, ret[" << ret << "]";
  }
  thread_local_rt_context = defaultDeviceContextMap_[device_id];
}

void AscendHalManager::InitializeAcl() {
}

bool AscendHalManager::EnableLccl() { return false; }
}  // namespace ascend
}  // namespace device
}  // namespace mrt
