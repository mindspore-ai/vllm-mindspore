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

#ifndef INFERRT_INFERRT_SRC_OPS_ASCEND_CUSTOM_UTILS_H_
#define INFERRT_INFERRT_SRC_OPS_ASCEND_CUSTOM_UTILS_H_

#include <torch/extension.h>
#include "ir/tensor/tensor.h"
#include "ir/value/value.h"
#include "common/visible.h"
namespace mrt::ops {

MRT_EXPORT at::Tensor ToTorchTensor(const ir::TensorPtr &tensor);
MRT_EXPORT ir::TensorPtr FromTorchTensor(const at::Tensor &tensor, bool isFake = false);
MRT_EXPORT void CheckOutputInputRef(const std::vector<const ir::Value *> &input, const ir::Value *output,
                                    const std::string &opName);
MRT_EXPORT bool IsTorchTensorStandardLayout(const at::Tensor &tensor);
}  // namespace mrt::ops
#endif  // INFERRT_INFERRT_SRC_OPS_ASCEND_CUSTOM_UTILS_H_
