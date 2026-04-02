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

#include "runtime/executor/kernel_capture/utils/graph_capture_manager.h"

namespace mrt {
namespace runtime {

void GraphCaptureManager::Initialize(const std::vector<OpRunner> &opRunners,
                                     const device::DeviceContext *expected_device_context) {
  (void)opRunners;
  (void)expected_device_context;
}

void GraphCaptureManager::CreateCaptureGraph(const device::DeviceContext *device_context, bool fullGraphMode) {
  (void)device_context;
  (void)fullGraphMode;
  if (!shape_key_.empty()) {
    (void)captured_shape_keys_.insert(shape_key_);
  }
}

bool GraphCaptureManager::LaunchAllKernelsWithCapture(std::vector<OpRunner> &opRunners) {
  (void)opRunners;
  return true;
}

bool GraphCaptureManager::LaunchAllKernelsWithCaptureFullGraph(std::vector<OpRunner> &opRunners, void *captureStream) {
  (void)opRunners;
  (void)captureStream;
  return true;
}

bool GraphCaptureManager::LaunchAllKernelsWithReplayFullGraph(std::vector<OpRunner> &opRunners, void *executeStream,
                                                              void *updateStream) {
  (void)opRunners;
  (void)executeStream;
  (void)updateStream;
  return true;
}

bool GraphCaptureManager::HasCapturedGraph() const {
  if (shape_key_.empty()) {
    return false;
  }
  return captured_shape_keys_.count(shape_key_) != 0;
}

}  // namespace runtime
}  // namespace mrt
