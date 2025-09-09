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

#include "runtime/builder/pipeline/pipeline_builder.h"
#include "runtime/executor/pipeline/pipeline_executor.h"

namespace mrt {
namespace runtime {
PipelineBuilder::PipelineBuilder(const ir::GraphPtr &graph) : Builder(graph) {}

std::unique_ptr<Executor> PipelineBuilder::BuildExecutor() {
  LOG_OUT << "Begin build pipeline executor.";
  std::unordered_map<ir::Node *, std::vector<size_t>> tensorFreePoint;
  // 1. Analyse ref count
  RecordTensorFreePoint(&tensorFreePoint);
  // 2. Creater OpRunner
  CreateOpRunners(&tensorFreePoint);

  auto pipelineExecutor = std::make_unique<PipelineExecutor>(opRunners_);
  pipelineExecutor->Initialize();
  LOG_OUT << "End build pipeline executor.";
  return pipelineExecutor;
}
}  // namespace runtime
}  // namespace mrt
