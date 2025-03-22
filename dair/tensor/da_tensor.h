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

#ifndef DA_TENSOR_MAX_INPUT
#define DA_TENSOR_MAX_INPUT 10
#endif
#ifndef DA_TENSOR_MAX_DIM
#define DA_TENSOR_MAX_DIM 6
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
// Data type of tensor
enum Type {
  Type_F16,
  Type_F32,
  Type_F64,
  Type_I16,
  Type_I32,
  Type_I64,
  Type_BF16,
  Type_End
};

struct DATensor {
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
  struct DATensor *input[DA_TENSOR_MAX_INPUT] = {nullptr};
};

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

DAContext *NewDAContext(size_t deviceId = 0, size_t memSize = kMemPoolSize);
void FreeDAContext(DAContext *context);

DAGraph *NewDAGraph(DAContext *context);
void AddParameter(DAGraph *graph, DATensor *param);

DATensor *NewDATensor(DAContext *context);
DATensor *NewDATensor(DAContext *context, Type type, size_t dim = 0,
                      size_t shape[DA_TENSOR_MAX_DIM] = nullptr,
                      void *data = nullptr, ops::Op op = ops::Op_End,
                      DATensor *input[DA_TENSOR_MAX_INPUT] = nullptr);
} // namespace tensor

#endif // __TENSOR_DA_TENSOR_H__