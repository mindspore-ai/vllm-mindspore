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

#ifndef __TENSOR_TENSOR_H__
#define __TENSOR_TENSOR_H__

#include <utility>
#include <string>

#include "common/visible.h"
#include "ops/op_def/ops_name.h"
#include "ir/tensor/tensor_data.h"

#ifndef DA_TENSOR_MAX_INPUT
#define DA_TENSOR_MAX_INPUT 512
#endif

#ifndef DA_GRAPH_MAX_PARAM
#define DA_GRAPH_MAX_PARAM 1024
#endif

#ifndef DA_GRAPH_MAX_NODE
#define DA_GRAPH_MAX_NODE 4096
#endif

#ifndef DA_CONTEXT_MAX_NUM
#define DA_CONTEXT_MAX_NUM 5
#endif

namespace da {
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
  // Device type of tensor
  TensorType tensorType{UNKNOW_TENSOR};

  // Data type of tensor
  Type type{Type_End};

  // Tensor data
  void *data{nullptr};

  // Number of dimensions
  size_t dim{0};
  // Shape of dimensions
  size_t shape[DA_TENSOR_MAX_DIM] = {0};

  // Operation of this tensor
  ops::Op op{ops::Op_End};
  // Inputs size
  size_t inputSize{0};
  // Input tensors
  TensorArrayPtr input{nullptr};
};

inline std::string ToString(const DATensor *tensor) {
  // CHECK_IF_NULL(tensor);
  std::stringstream ss;
  ss << "tensor{";
  ss << ops::ToStr(tensor->op);
  ss << ", shape: [";
  for (size_t i = 0; i < tensor->dim; ++i) {
    ss << tensor->shape[i] << (i == tensor->dim - 1 ? "" : ", ");
  }
  ss << "], ptr: " << tensor;
  ss << "}";
  return ss.str();
}

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
DATensor *NewDATensor(DAContext *context, Type type, size_t dim = 0, const ShapeArray &shape = {0},
                      void *data = nullptr, ops::Op op = ops::Op_End, size_t inputSize = 0,
                      const TensorArrayPtr &input = {nullptr});
// Create a new DATensorList with given length
DATensor **NewDATensorList(DAContext *context, size_t len);

// directly use the original data address, no copy
template <typename T>
TensorData *NewTensorData(DAContext *ctx, Type dtype, const ShapeArray &shape, void *data) {
  CHECK_IF_NULL(ctx);
  CHECK_IF_NULL(ctx->memPool);

  size_t tensorDataSize = sizeof(TensorDataImpl<T>);
  auto newSize = ctx->memUsed + tensorDataSize;
  CHECK_IF_FAIL(newSize < ctx->memSize);
  TensorDataImpl<T> *tensorData = (TensorDataImpl<T> *)(reinterpret_cast<char *>(ctx->memPool) + ctx->memUsed);
  ctx->memUsed = newSize;
  tensorData->ndim = ShapeDims(shape);
  tensorData->size = ShapeSize(shape);
  tensorData->nbytes = tensorData->size * sizeof(T);
  tensorData->data = static_cast<T *>(data);
  return tensorData;
}

template <typename... Args>
TensorData *MakeTensorData(DAContext *ctx, Type dtype, Args &&... args) {
  switch (dtype) {
    case Type_Bool:
      return NewTensorData<bool>(ctx, dtype, std::forward<Args>(args)...);
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
}  // namespace tensor
}  // namespace da

#endif  // __TENSOR_TENSOR_H__
