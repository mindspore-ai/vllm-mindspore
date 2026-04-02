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

#ifndef __RUNTIME_BUILDER_KERNEL_CAPTURE_BUILDER_H__
#define __RUNTIME_BUILDER_KERNEL_CAPTURE_BUILDER_H__

#include "runtime/builder/builder.h"
#include "runtime/executor/kernel_capture/utils/graph_capture_manager.h"

namespace mrt {
namespace runtime {
class DA_API KernelCaptureBuilder : public Builder {
 public:
  KernelCaptureBuilder() = delete;
  explicit KernelCaptureBuilder(const ir::GraphPtr &graph);
  ~KernelCaptureBuilder() override = default;

  std::unique_ptr<Executor> BuildExecutor() override;

  // Get the op runners created during analysis
  std::shared_ptr<std::vector<OpRunner>> GetOpRunners() const { return this->opRunners_; }

 private:
  // Store analysis results
  std::vector<std::pair<size_t, size_t>> capture_kernel_range_positions_;
  std::vector<std::pair<ExecutorType, size_t>> executors_;
};
}  // namespace runtime
}  // namespace mrt

#endif  // __RUNTIME_BUILDER_KERNEL_CAPTURE_BUILDER_H__
