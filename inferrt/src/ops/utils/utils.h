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

#ifndef __OPS_OP_BASE_OP_UTILS_H__
#define __OPS_OP_BASE_OP_UTILS_H__

#include <vector>

#include "common/common.h"
#include "ir/tensor/tensor.h"
#include "ir/tensor/format.h"

namespace mrt {
namespace ops {
using ir::MemoryFormat;
/// Cache capacity for ATB; from env MS_INFERRT_ATB_CACHE_CAPACITY, default 64.
MRT_EXPORT size_t GetAtbCacheCapacity();
/// Cache capacity for ACLNN; from env MS_INFERRT_ACLNN_CACHE_CAPACITY, default 10000.
MRT_EXPORT size_t GetAclnnCacheCapacity();
MRT_EXPORT void CalBroadCastShape(const std::vector<int64_t> &xShape, const std::vector<int64_t> &yShape,
                                  std::vector<int64_t> *broadcastShape);
MRT_EXPORT bool IsBaseFormat(MemoryFormat format);
MRT_EXPORT bool IsTensorBaseFormat(const ir::TensorPtr &tensor);
}  // namespace ops
}  // namespace mrt

#endif  // __OPS_OP_BASE_OP_UTILS_H__
