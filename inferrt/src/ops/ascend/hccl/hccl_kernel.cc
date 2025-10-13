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

#include "ops/ascend/hccl/hccl_kernel.h"

#include <map>
#include <set>
#include <unordered_set>

#include "ops/ascend/hccl/hccl_adapter.h"
#include "ops/ascend/hccl/hcom_utils.h"

namespace mrt {
namespace ops {

HcclKernel::HcclKernel() : hccl_count_(0), root_id_(0), src_rank_(0), dest_rank_(0), comm_(nullptr) {}

}  // namespace ops
}  // namespace mrt
