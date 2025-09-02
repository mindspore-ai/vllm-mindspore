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

#ifndef OPS_ASCEND_HCCL_KERNEL_H_
#define OPS_ASCEND_HCCL_KERNEL_H_

#include <map>
#include <memory>
#include <string>
#include <vector>
#include <algorithm>
#include <utility>
#include <condition_variable>

#include "ops/operator.h"
#include "ops/ascend/hccl/hcom_utils.h"
#include "hccl/hcom.h"
#include "hccl/hccl_types.h"

namespace mrt {
namespace ops {
class HcclKernel {
 public:
  HcclKernel();
  ~HcclKernel() = default;

 public:
  HcclDataType hccl_data_type_;
  uint64_t hccl_count_;
  uint32_t root_id_;
  uint32_t src_rank_;
  uint32_t dest_rank_;
  std::string group_;
  HcclComm comm_;
  ulong loop_size_{0};
  bool is_graph_mode_{false};
  std::string hccl_inner_comm_name_;
};

}  // namespace ops
}  // namespace mrt
#endif  // OPS_ASCEND_HCCL_KERNEL_H_
