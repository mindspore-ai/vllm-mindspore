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

#include "hardware/ascend/ascend_device_context.h"
#include <tuple>
#include <algorithm>
#include <sstream>
#include <map>
#include <thread>
#include <set>
#include <unistd.h>
#include "hardware/hardware_abstract/device_context_manager.h"
#include "hardware/ascend/res_manager/symbol_interface/acl_base_symbol.h"
#include "hardware/ascend/res_manager/symbol_interface/acl_rt_symbol.h"
#include "hardware/ascend/res_manager/symbol_interface/symbol_utils.h"
#include "hardware/ascend/res_manager/symbol_interface/acl_compiler_symbol.h"
#include "common/common.h"

namespace mrt {
namespace device {
namespace ascend {
namespace {
constexpr auto kSaturationMode = "Saturation";
constexpr auto kINFNANMode = "INFNAN";
const char kAscendDevice[] = "Ascend";
}  // namespace

void AscendDeviceContext::InitializeForAclop() const {
  if (initializedAclop_) {
    return;
  }

  LOG_OUT << "Start initializing for acl.";
  LoadAscendApiSymbols();

  LOG_OUT << "End initializing for acl.";
}

void AscendDeviceContext::Initialize() {
  std::lock_guard<std::mutex> lock(initMutex_);
  if (initialized_) {
    return;
  }

  LOG_OUT << "Start initializing device context.";
  LoadAscendApiSymbols();

  CHECK_IF_NULL(deviceResManager_);
  deviceResManager_->Initialize();

  initialized_ = true;
  pid_ = getpid();  // set the pid when first initialize
  LOG_OUT << "End initializing device context.";
}

void AscendDeviceContext::Destroy() {
  if (pid_ != getpid()) {
    // Check whether the device context needs to be released.
    // The device context is copied by the dataset independent process, but does not need to be released
    // in the dataset independent process.
    // The device context is copied from main process by fork
    LOG_OUT << "The device context is not initialized by current process, it doesn't need to be destroyed.";
    return;
  }

  if (deviceResManager_ == nullptr) {
    return;
  }
  // Device resource manager must be destroyed before 'FinalizeGe' unless some runtime APIs will throw exception.
  // for ge, has destropy in graph_executor->finalize
  deviceResManager_->Destroy();

  initialized_ = false;
}

MS_REGISTER_DEVICE(kAscendDevice, AscendDeviceContext);
}  // namespace ascend
}  // namespace device
}  // namespace mrt
