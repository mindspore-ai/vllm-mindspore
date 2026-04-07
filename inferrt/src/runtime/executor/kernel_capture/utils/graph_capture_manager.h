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

#ifndef __RUNTIME_EXECUTOR_KERNEL_CAPTURE_UTILS_GRAPH_CAPTURE_MANAGER_H__
#define __RUNTIME_EXECUTOR_KERNEL_CAPTURE_UTILS_GRAPH_CAPTURE_MANAGER_H__

#include <string>
#include <unordered_set>
#include <vector>

#include "runtime/executor/op_runner.h"
#include "hardware/hardware_abstract/device_context.h"

namespace mrt {
namespace runtime {

enum ExecutorType {
  KERNEL = 0,
  CAPTURE_GRAPH = 1,
};

class DA_API GraphCaptureManager {
 public:
  GraphCaptureManager() = default;
  ~GraphCaptureManager() = default;

  void Initialize(const std::vector<OpRunner> &opRunners, const device::DeviceContext *expected_device_context);
  void CreateCaptureGraph(const device::DeviceContext *device_context, bool fullGraphMode);
  bool LaunchAllKernelsWithCapture(std::vector<OpRunner> &opRunners);
  bool LaunchAllKernelsWithCaptureFullGraph(std::vector<OpRunner> &opRunners, void *captureStream);
  bool LaunchAllKernelsWithReplayFullGraph(std::vector<OpRunner> &opRunners, void *executeStream, void *updateStream);

  void SetShapeKey(const std::string &shape_key) { shape_key_ = shape_key; }
  bool HasCapturedGraph() const;

 private:
  std::string shape_key_;
  std::unordered_set<std::string> captured_shape_keys_;
};

}  // namespace runtime
}  // namespace mrt

#endif  // __RUNTIME_EXECUTOR_KERNEL_CAPTURE_UTILS_GRAPH_CAPTURE_MANAGER_H__
