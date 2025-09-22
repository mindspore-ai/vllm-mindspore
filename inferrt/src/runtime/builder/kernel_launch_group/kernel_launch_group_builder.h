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

#ifndef __RUNTIME_BUILDER_KERNEL_LAUNCH_GROUP_BUILDER_H__
#define __RUNTIME_BUILDER_KERNEL_LAUNCH_GROUP_BUILDER_H__

#include <vector>
#include <memory>
#include <unordered_set>
#include <utility>
#include "runtime/builder/builder.h"

namespace mrt {
namespace runtime {
class DA_API KernelLaunchGroupBuilder : public Builder {
 public:
  explicit KernelLaunchGroupBuilder(const ir::GraphPtr &graph);
  ~KernelLaunchGroupBuilder() override = default;

  std::unique_ptr<Executor> BuildExecutor() override;

 private:
  void CheckGroupLaunchRequirements() const;
  void PartitionKernelLaunchGroups();
  void RecordDynamicInputs();
  void RecordGraphOutputs();

  uint64_t parallelDispatchNum_;
  uint64_t parallelSliceNum_;
  std::shared_ptr<std::vector<std::pair<OpRunner *, size_t>>> opRunnerGroups_;
  std::shared_ptr<std::vector<OpRunner *>> serialLaunchOps_;
  std::shared_ptr<std::vector<std::pair<ir::TensorPtr, std::vector<int64_t>>>> graphInputsWithShape_;
  std::shared_ptr<std::unordered_set<ir::Tensor *>> graphOutputs_;
};

}  // namespace runtime
}  // namespace mrt

#endif  // __RUNTIME_BUILDER_KERNEL_LAUNCH_GROUP_BUILDER_H__
