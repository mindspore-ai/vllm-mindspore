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

#ifndef __RUNTIME_EXECUTOR_H__
#define __RUNTIME_EXECUTOR_H__

#include "common/common.h"
#include "tensor/da_tensor.h"

#include <unordered_map>
#include <vector>

#define DEBUG

namespace runtime {
using namespace tensor;

class GraphExecutor {
public:
  GraphExecutor() : context_{NewDAContext()} { CHECK_NULL(context_); }
  ~GraphExecutor() {
    CHECK_NULL(context_);
    FreeDAContext(context_);
  }

  void BeginGraph();
  void EndGraph();
  DATensor *AddTensor(Type type = Type_F32, size_t dim = 0,
                      size_t shape[DA_TENSOR_MAX_DIM] = nullptr,
                      void *data = nullptr);
  DATensor *AddTensor(const std::vector<DATensor *> &inputs);

  void Run();
  void RunTensor(const DATensor *tensor);

private:
  DAContext *context_{nullptr};
  DAGraph *graph_{nullptr};
#ifdef DEBUG
  std::unordered_map<const DATensor *, size_t> tensorNumMap_;
#endif
};
} // namespace runtime

#endif // __RUNTIME_EXECUTOR_H__