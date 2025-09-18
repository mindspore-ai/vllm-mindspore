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

#ifndef __RUNTIME_BUILDER_BUILDER_H__
#define __RUNTIME_BUILDER_BUILDER_H__

#include <vector>
#include <unordered_map>

#include "ops/operator.h"
#include "runtime/executor/op_runner.h"

namespace mrt {
namespace runtime {
class Executor;

/**
 * @brief Base class for building executor.
 *
 * The Builder class is responsible for constructing executor that can run
 * computational graph. It analyzes tensor dependencies, creates operator runners,
 * and builds the final executor with all necessary components.
 */
class DA_API Builder {
 public:
  Builder() = delete;
  Builder(const ir::GraphPtr &graph) : graph_(graph) {}
  virtual ~Builder() = default;

  /**
   * @brief Build an executor for the computational graph.
   *
   * This method orchestrates the building process by:
   * 1. Analyzing tensor reference count
   * 2. Creating operation runners
   * 3. Constructing the final executor
   *
   * @return A unique pointer to the constructed executor.
   */
  virtual std::unique_ptr<Executor> BuildExecutor();

 protected:
  /**
   * @brief Records storage free points to optimize memory management.
   *
   * This method analyzes the computational graph to determine when storages
   * are no longer needed and can be freed to optimize memory usage.
   */
  void RecordStorageFreePoint();

  /**
   * @brief Creates operation runners for all nodes in the graph.
   *
   * This method iterates through all nodes in the computational graph,
   * creates corresponding operator runners, and configures them with
   * appropriate memory management settings.
   */
  void CreateOpRunners();

  // The graph that the executor will run.
  ir::GraphPtr graph_{nullptr};

  // Shared pointer to the vector of OpRunners for all operators by execution order in graph_.
  std::shared_ptr<std::vector<OpRunner>> opRunners_{nullptr};

  // The storages that can be safely freed once the last consumer node is done.
  std::unordered_map<ir::Node *, std::vector<ir::Storage *>> storagesToFree_;
};
}  // namespace runtime
}  // namespace mrt

#endif  // __RUNTIME_BUILDER_BUILDER_H__
