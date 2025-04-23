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

#ifndef __TENSOR_DA_TENSOR_H__
#define __TENSOR_DA_TENSOR_H__

#include <stdlib.h>

#include "ops/ops_name.h"
#include "tensor/tensor_data.h"

#ifndef DA_TENSOR_MAX_INPUT
#define DA_TENSOR_MAX_INPUT 10
#endif

#ifndef DA_GRAPH_MAX_PARAM
#define DA_GRAPH_MAX_PARAM 64
#endif

#ifndef DA_GRAPH_MAX_NODE
#define DA_GRAPH_MAX_NODE 4096
#endif

#ifndef DA_CONTEXT_MAX_NUM
#define DA_CONTEXT_MAX_NUM 5
#endif

namespace tensor {
struct DATensor;
using TensorArrayPtr = DATensor *[DA_TENSOR_MAX_INPUT];

struct DAContext {
  // Device Id
  int deviceId{-1};

  // Memory pool size
  size_t memSize{0};
  // Used size of memory pool
  size_t memUsed{0};
  // Memory pool
  void *memPool{nullptr};
};

struct DAContextManager {
  // Context used state
  bool used[DA_CONTEXT_MAX_NUM];

  // Contexts
  DAContext context[DA_CONTEXT_MAX_NUM];
};

constexpr auto kMemPoolSize = 1024 * 1024 * 256;

struct DAGraph {
  // Size of parameters
  size_t paramSize{0};
  // Tensor parameters
  DATensor *param[DA_GRAPH_MAX_PARAM] = {nullptr};

  // Size of nodes
  size_t nodeSize{0};
  // Tensor nodes
  DATensor *node[DA_GRAPH_MAX_NODE] = {nullptr};
};

struct DATensor {
  // Data type of tensor
  Type type{Type_End};
  // tensor type of this tensor
  TensorType tensor_type{HOST_TENSOR};
  // tensor shape
  ShapeArrayPtr shape{0};
  // tensor data
  TensorData *data{nullptr};
  // Operation of this tensor
  ops::Op op{ops::Op_End};
  // Inputs size
  size_t inputs_size{0};
  // Input tensors
  TensorArrayPtr inputs{nullptr};
};

// Create a new DAContext.
DAContext *NewDAContext(size_t deviceId = 0, size_t memSize = kMemPoolSize);
// Free the DAContext.
void FreeDAContext(DAContext *context);

// Create a new DAGraph.
DAGraph *NewDAGraph(DAContext *context);
// Add a parameter for DAGraph.
void AddParameter(DAGraph *graph, DATensor *param);
// Add a tensor for DAGraph.
void AddTensor(DAGraph *graph, DATensor *tensor);

// Create a new DATensor.
DATensor *NewDATensor(DAContext *context);
// Create a new DATensor.
DATensor *NewDATensor(DAContext *context, Type type, size_t dim = 0,
                      const ShapeArrayPtr &shape = {0}, void *data = nullptr,
                      ops::Op op = ops::Op_End,
                      const TensorArrayPtr &inputs = {nullptr});

// directly use the original data address, no copy
template <typename T>
TensorData *NewTensorData(DAContext *ctx, Type dtype,
                          const ShapeArrayPtr &shape, void *data) {
  CHECK_IF_NULL(ctx);
  CHECK_IF_NULL(ctx->memPool);

  size_t tensor_data_size = sizeof(TensorDataImpl<T>);
  auto new_size = ctx->memUsed + tensor_data_size;
  CHECK_IF_FAIL(new_size < ctx->memSize);
  TensorDataImpl<T> *tensor_data = (TensorDataImpl<T> *)((char *)ctx->memPool + ctx->memUsed);
  ctx->memUsed = new_size;
  tensor_data->ndim = ShapeDims(shape);
  tensor_data->size = ShapeSize(shape);
  tensor_data->nbytes = tensor_data->size * sizeof(T);
  tensor_data->data = static_cast<T *>(data);
  return tensor_data;
}

template <typename... Args>
TensorData *MakeTensorData(DAContext *ctx, Type dtype, Args &&...args) {
  switch (dtype) {
  case Type_I16:
    return NewTensorData<int16_t>(ctx, dtype, std::forward<Args>(args)...);
  case Type_I32:
    return NewTensorData<int32_t>(ctx, dtype, std::forward<Args>(args)...);
  case Type_I64:
    return NewTensorData<int64_t>(ctx, dtype, std::forward<Args>(args)...);
  case Type_F32:
    return NewTensorData<float>(ctx, dtype, std::forward<Args>(args)...);
  case Type_F64:
    return NewTensorData<double>(ctx, dtype, std::forward<Args>(args)...);
  // todo support bf16 and fp16
  default:
    break;
  }
  LOG_OUT << "can not construct tensor data from unsupported dtype.";
  return nullptr;
}
} // namespace tensor

#endif // __TENSOR_DA_TENSOR_H__