/**
 * Copyright 2025 Zhang Qinghua
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

#include "tensor/da_tensor.h"
#include "common/common.h"

namespace tensor {
DAContext *NewDAContext() {
  static DAContextManager _manager;
  for (size_t i = 0; i < DA_CONTEXT_MAX_NUM; ++i) {
    if (!_manager.used[i]) {
      _manager.used[i] = true;
      return &_manager.context[i];
    }
  }
  return nullptr;
}

DAGraph *NewDAGraph(DAContext *context) {
  CHECK_NULL(context);
  CHECK_NULL(context->memPool);

  constexpr size_t graphSize = sizeof(DAGraph);
  auto newSize = context->memUsed + graphSize;
  CHECK_FAIL(newSize < context->memSize);

  DAGraph *graph = (DAGraph *)((char *)context->memPool + context->memUsed);
  context->memUsed = newSize;
  return graph;
}

DATensor *NewDATensor(DAContext *context) {
  CHECK_NULL(context);
  CHECK_NULL(context->memPool);

  constexpr size_t tensorSize = sizeof(DATensor);
  auto newSize = context->memUsed + tensorSize;
  CHECK_FAIL(newSize < context->memSize);

  DATensor *tensor = (DATensor *)((char *)context->memPool + context->memUsed);
  context->memUsed = newSize;
  return tensor;
}

DATensor *NewDATensor(DAContext *context, Type type, void *data, size_t dim,
                      size_t shape[DA_TENSOR_MAX_DIM], Op op,
                      DATensor *input[DA_TENSOR_MAX_INPUT]) {
  CHECK_NULL(context);
  CHECK_NULL(context->memPool);

  constexpr size_t tensorSize = sizeof(DATensor);
  auto newSize = context->memUsed + tensorSize;
  CHECK_FAIL(newSize < context->memSize);

  DATensor *tensor = (DATensor *)((char *)context->memPool + context->memUsed);
  context->memUsed = newSize;
  *tensor = (DATensor){type, data, dim, {0}, op, {0}};
  if (shape != nullptr) {
    for (size_t i = 0; i < DA_TENSOR_MAX_DIM; ++i) {
      tensor->shape[i] = shape[i];
    }
  }
  if (input != nullptr) {
    for (size_t i = 0; i < DA_TENSOR_MAX_INPUT; ++i) {
      tensor->input[i] = input[i];
    }
  }
  return tensor;
}
} // namespace tensor