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

#ifndef __RUNTIME_EXECUTOR_PIPELINE_EXECUTOR_H__
#define __RUNTIME_EXECUTOR_PIPELINE_EXECUTOR_H__

#include "runtime/executor/executor.h"

namespace mrt {
namespace runtime {
class DA_API PipelineExecutor : public Executor {
 public:
  PipelineExecutor() = default;
  ~PipelineExecutor() override = default;

  void Run(bool isDynamic) override;
};

}  // namespace runtime
}  // namespace mrt

#endif  // __RUNTIME_EXECUTOR_PIPELINE_EXECUTOR_H__
