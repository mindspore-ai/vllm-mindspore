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

#include "runtime/builder/kernel_capture/kernel_capture_builder.h"
#include "runtime/executor/kernel_capture/kernel_capture_executor.h"
#include "runtime/executor/kernel_capture/utils/graph_capture_manager.h"
#include "ir/graph.h"

namespace mrt {
namespace runtime {

KernelCaptureBuilder::KernelCaptureBuilder(const ir::GraphPtr &graph) : Builder(graph) {}

std::unique_ptr<Executor> KernelCaptureBuilder::BuildExecutor() {
  LOG_OUT << "Begin build kernel capture executor.";

  // Setup OpRunners for the base graph
  SetupOpRunners();
  auto kernelCaptureExecutor = std::make_unique<KernelCaptureExecutor>(opRunners_, deviceContexts_, GetGraphOutput());

  // Initialize the executor
  kernelCaptureExecutor->Initialize(graph_);

  LOG_OUT << "End build kernel capture executor.";
  return kernelCaptureExecutor;
}

}  // namespace runtime
}  // namespace mrt
