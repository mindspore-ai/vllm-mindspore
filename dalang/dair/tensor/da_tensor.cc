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

#undef DEBUG
#ifndef DEBUG
#undef LOG_OUT
#define LOG_OUT NO_LOG_OUT
#endif

namespace tensor {
DAContext *NewDAContext(size_t deviceId, size_t memSize) {
  static DAContextManager _manager;
  for (size_t i = 0; i < DA_CONTEXT_MAX_NUM; ++i) {
    if (!_manager.used[i]) {
      _manager.used[i] = true;
      auto &context = _manager.context[i];
      context.deviceId = deviceId;
      context.memSize = memSize;
      context.memPool = malloc(memSize);
      LOG_OUT << "Create a new DAContext " << &context
              << ", pool size: " << memSize;
      return &context;
    }
  }
  LOG_ERROR << "Failed to create new DAContext";
  return nullptr;
}

void FreeDAContext(DAContext *context) {
  if (context != nullptr && context->memPool != nullptr) {
    context->deviceId = -1;
    context->memSize = 0;
    free(context->memPool);
    context->memPool = nullptr;
    LOG_OUT << "Free DAContext " << context;
  } else {
    LOG_ERROR << "context is null or context.memPool is null";
    exit(EXIT_FAILURE);
  }
}

DAGraph *NewDAGraph(DAContext *context) {
  CHECK_IF_NULL(context);
  CHECK_IF_NULL(context->memPool);

  constexpr size_t graphSize = sizeof(DAGraph);
  auto newSize = context->memUsed + graphSize;
  CHECK_IF_FAIL(newSize < context->memSize);

  DAGraph *graph = (DAGraph *)((char *)context->memPool + context->memUsed);
  context->memUsed = newSize;
  LOG_OUT << "Create DAGraph " << graph << ", size: " << sizeof(DAGraph)
          << ", for DAContext " << context;

  return graph;
}

void AddParameter(DAGraph *graph, DATensor *param) {
  CHECK_IF_NULL(graph);
  CHECK_IF_FAIL(graph->paramSize < DA_GRAPH_MAX_PARAM);
  graph->param[graph->paramSize] = param;
  ++graph->paramSize;
  LOG_OUT << "Add Parameter " << param << " for DAGraph " << graph;
}

void AddTensor(DAGraph *graph, DATensor *tensor) {
  CHECK_IF_NULL(graph);
  CHECK_IF_NULL(tensor);
  CHECK_IF_FAIL(graph->nodeSize < DA_GRAPH_MAX_NODE);
  graph->node[graph->nodeSize] = tensor;
  ++graph->nodeSize;
  LOG_OUT << "Add Tensor " << tensor << " for DAGraph " << graph;
}

DATensor *NewDATensor(DAContext *context) {
  CHECK_IF_NULL(context);
  CHECK_IF_NULL(context->memPool);

  constexpr size_t tensorSize = sizeof(DATensor);
  auto newSize = context->memUsed + tensorSize;
  CHECK_IF_FAIL(newSize < context->memSize);

  DATensor *tensor = (DATensor *)((char *)context->memPool + context->memUsed);
  context->memUsed = newSize;
  tensor->dim = 0;
  tensor->shape[0] = 0;
  tensor->type = Type_F32;
  LOG_OUT << "Create DATensor " << tensor << ", size: " << sizeof(DATensor)
          << ", for DAContext " << context;
  return tensor;
}

DATensor *NewDATensor(DAContext *context, Type type, size_t dim,
                      size_t shape[DA_TENSOR_MAX_DIM], void *data, ops::Op op,
                      DATensor *input[DA_TENSOR_MAX_INPUT]) {
  CHECK_IF_NULL(context);
  CHECK_IF_NULL(context->memPool);

  constexpr size_t tensorSize = sizeof(DATensor);
  auto newSize = context->memUsed + tensorSize;
  CHECK_IF_FAIL(newSize < context->memSize);

  DATensor *tensor = (DATensor *)((char *)context->memPool + context->memUsed);
  context->memUsed = newSize;
  *tensor = (DATensor){type, data, dim, {0}, op, 0, {0}};
  if (shape != nullptr) {
    for (size_t i = 0; i < DA_TENSOR_MAX_DIM; ++i) {
      tensor->shape[i] = shape[i];
      if (shape[i] == 0) {
        break;
      }
    }
  }
  if (input != nullptr) {
    for (size_t i = 0; i < DA_TENSOR_MAX_INPUT; ++i) {
      tensor->input[i] = input[i];
      if (input[i] == nullptr) {
        break;
      }
    }
  }
  LOG_OUT << "Create DATensor " << tensor << ", size: " << sizeof(DATensor)
          << ", for DAContext " << context;
  return tensor;
}
} // namespace tensor