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

#ifndef __RUNTIME_UTILS_SHAPE_KEY_UTILS_H__
#define __RUNTIME_UTILS_SHAPE_KEY_UTILS_H__

#include <string>
#include <sstream>
#include <vector>
#include "ir/graph.h"

namespace mrt {
namespace runtime {

/**
 * @brief Generate a shape key string from graph runtime inputs.
 *
 * This function generates a unique key based on the shapes of tensors in the graph runtime inputs,
 * which can be used for caching AclGraphs for different input shapes.
 *
 * @param graph The graph containing runtime inputs
 * @return std::string The generated shape key
 */
inline std::string GenerateShapeKey(const ir::GraphPtr &graph) {
  std::stringstream ss;
  bool first = true;

  const auto &shapeNodes = graph->inputs.empty() ? graph->parameters : graph->inputs;
  for (const auto &inputNode : shapeNodes) {
    if (!inputNode || !inputNode->output) {
      continue;
    }

    // Visit all tensors in the input output and extract their shapes
    ir::VisitAllTensors(inputNode->output, [&](const ir::TensorPtr &tensor) {
      if (!tensor) {
        return;
      }

      const auto &shape = tensor->Shape();
      for (size_t i = 0; i < shape.size(); ++i) {
        if (!first) {
          ss << "-";
        }
        ss << shape[i];
        first = false;
      }
    });
  }

  return ss.str();
}

}  // namespace runtime
}  // namespace mrt

#endif  // __RUNTIME_UTILS_SHAPE_KEY_UTILS_H__
