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

#include "rt/runtime/kernel.h"
#include "runtime/utils.h"

namespace da {
namespace runtime {
void DAKernel::RunKernel(bool isDynamic) {
  CHECK_IF_NULL(tensorNode_);
  if (isDynamic) {
    InferShape();
    Resize();
  } else if (IsDAKernelNeedForceResize(tensorNode_)) {
    Resize();
  }

  if (IsDAKernelSkipLaunch(tensorNode_)) {
    LOG_OUT << "Skip launch kernel for ops." << ops::ToStr(tensorNode_->op);
    tensorNode_->data = tensorNode_->input[GetDATensorOutputValueFromInputIndex(tensorNode_)]->data;
    return;
  }

  Launch();
}
}  // namespace runtime
}  // namespace da
