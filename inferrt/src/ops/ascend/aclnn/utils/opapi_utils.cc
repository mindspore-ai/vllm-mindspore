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

#include <set>
#include <unordered_map>

#include "common/logger.h"
#include "config/device/ascend/op_precision_conf.h"
#include "ir/tensor/format.h"
#include "ops/ascend/aclnn/utils/opapi_utils.h"

namespace mrt {
namespace ops {
namespace {
using config::ascend::AclCubeMathType;
using config::ascend::OpPrecisionConf;
using ir::MemoryFormat;
constexpr auto AclCubeMathTypeArraySize = 4;

constexpr AclCubeMathType AclCubeMathTypeArray[AclCubeMathTypeArraySize] = {
  AclCubeMathType::KEEP_DTYPE,
  AclCubeMathType::USE_FP16,
  AclCubeMathType::USE_HF32,
  AclCubeMathType::ALLOW_FP32_DOWN_PRECISION,
};

const std::set<MemoryFormat> BaseFormatSet = {
  MemoryFormat::FORMAT_ND,
  MemoryFormat::FORMAT_NCHW,
  MemoryFormat::FORMAT_NHWC,
  MemoryFormat::FORMAT_NCDHW,
};
}  // namespace

int8_t GetCubeMathType() {
  auto &opPrecisionConf = OpPrecisionConf::Instance();
  uint8_t cubeMathTypeIndex = (static_cast<uint8_t>(opPrecisionConf.IsAllowMatmulHF32()) << 1) +
                              static_cast<uint8_t>(opPrecisionConf.IsAllowFP32ToFP16());
  if (cubeMathTypeIndex >= AclCubeMathTypeArraySize) {
    LOG_OUT << "Invalid cubeMathType index: " << cubeMathTypeIndex
            << ", set AclCubeMathType to ALLOW_FP32_DOWN_PRECISION";
    return static_cast<int8_t>(AclCubeMathType::ALLOW_FP32_DOWN_PRECISION);
  }
  return static_cast<int8_t>(AclCubeMathTypeArray[cubeMathTypeIndex]);
}

bool IsTensorBaseFormat(const ir::TensorPtr &tensor) { return BaseFormatSet.count(tensor->Format()) != 0; }

}  // namespace ops
}  // namespace mrt
