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

#include <vector>

#include "ops/ascend/aclnn/aclnn_add_rms_norm_quant_v2.h"
#include "ops/ascend/aclnn/utils/opapi_utils.h"
#include "ops/op_register.h"

namespace mrt {
namespace ops {
namespace {
inline std::optional<ir::TensorPtr> GetOptionalTensor(const ir::Value *value) {
  return value->IsTensor() ? std::optional(value->ToTensor()) : std::nullopt;
}
}  // namespace

// Input parameter index definitions aligned with frontend signature:
// npu_add_rms_norm_quant(x1, x2, gamma, scales1, zero_points1=None, beta=None,
//                        scales2=None, zero_points2=None, axis=-1, epsilon=1e-06, div_mode=True)
constexpr size_t kX1Idx = 0;
constexpr size_t kX2Idx = 1;
constexpr size_t kGammaIdx = 2;
constexpr size_t kScales1Idx = 3;
constexpr size_t kZeroPoints1OptionalIdx = 4;
constexpr size_t kBetaOptionalIdx = 5;
constexpr size_t kScales2OptionalIdx = 6;
constexpr size_t kZeroPoints2OptionalIdx = 7;
constexpr size_t kAxisIdx = 8;
constexpr size_t kEpsilonIdx = 9;
constexpr size_t kDivModeIdx = 10;

// Output parameter index definitions aligned with ACLNN interface: y1Out, y2Out, xOut, rmsNormOut
constexpr size_t kY1OutIdx = 0;
constexpr size_t kY2OutIdx = 1;
constexpr size_t kXOutIdx = 2;
constexpr size_t kRmsNormOutIdx = 3;

OpsErrorCode AclnnAddRmsNormQuantV2::CalcWorkspace(const std::vector<const ir::Value *> &input, const ir::Value *output,
                                                   size_t *workspaceSize) {
  LOG_OUT << "Begin CalcWorkspace for op [add_rms_norm_quant_v2]";

  // Parameter validation: scales2 must be None
  auto scales2_opt = GetOptionalTensor(input[kScales2OptionalIdx]);
  if (scales2_opt.has_value()) {
    LOG_OUT << "Error: scales2 only support None.";
    return INVALID_PARAM;
  }

  // Parameter validation: zero_points2 must be None
  auto zero_points2_opt = GetOptionalTensor(input[kZeroPoints2OptionalIdx]);
  if (zero_points2_opt.has_value()) {
    LOG_OUT << "Error: zero_points2 only support None.";
    return INVALID_PARAM;
  }

  // Parameter validation: axis must be -1
  auto axis_val = input[kAxisIdx]->ToInt();
  if (axis_val != -1) {
    LOG_OUT << "Error: axis only support -1, but got " << axis_val << ".";
    return INVALID_PARAM;
  }

  // Parameter validation: div_mode must be True
  auto div_mode_val = input[kDivModeIdx]->ToBool();
  if (!div_mode_val) {
    LOG_OUT << "Error: div_mode only support True.";
    return INVALID_PARAM;
  }

  auto &outputTuple = output->ToTuple();

  // Note: The ACLNN interface parameter order is:
  // x1, x2, gamma, scales1,
  // scales2Optional, zeroPoints1Optional, zeroPoints2Optional, betaOptional,
  // axis, epsilon, divMode, workspaceSize, executor
  executor_->GetWorkspaceSize(
    static_cast<uint64_t *>(workspaceSize), input[kX1Idx]->ToTensor(), input[kX2Idx]->ToTensor(),
    input[kGammaIdx]->ToTensor(), input[kScales1Idx]->ToTensor(),
    // Optional parameters must match ACLNN interface order: scales2, zeroPoints1, zeroPoints2, beta
    GetOptionalTensor(input[kScales2OptionalIdx]), GetOptionalTensor(input[kZeroPoints1OptionalIdx]),
    GetOptionalTensor(input[kZeroPoints2OptionalIdx]), GetOptionalTensor(input[kBetaOptionalIdx]),
    input[kAxisIdx]->ToInt(), input[kEpsilonIdx]->ToDouble(), input[kDivModeIdx]->ToBool(),
    (*outputTuple)[kY1OutIdx]->ToTensor(), (*outputTuple)[kY2OutIdx]->ToTensor(), (*outputTuple)[kXOutIdx]->ToTensor(),
    nullptr);

  return SUCCESS;
}

OpsErrorCode AclnnAddRmsNormQuantV2::Launch(const std::vector<const ir::Value *> &input, void *workspace,
                                            size_t workspaceSize, ir::Value *output, void *stream) {
  LOG_OUT << "Begin Launch for op [add_rms_norm_quant_v2]";

  auto &outputTuple = output->ToTuple();

  // Parameter order must match CalcWorkspace and ACLNN aclnnAddRmsNormQuantV2 interface
  executor_->Launch(workspace, workspaceSize, stream, input[kX1Idx]->ToTensor(), input[kX2Idx]->ToTensor(),
                    input[kGammaIdx]->ToTensor(), input[kScales1Idx]->ToTensor(),
                    // Optional parameters order: scales2, zeroPoints1, zeroPoints2, beta
                    GetOptionalTensor(input[kScales2OptionalIdx]), GetOptionalTensor(input[kZeroPoints1OptionalIdx]),
                    GetOptionalTensor(input[kZeroPoints2OptionalIdx]), GetOptionalTensor(input[kBetaOptionalIdx]),
                    input[kAxisIdx]->ToInt(), input[kEpsilonIdx]->ToDouble(), input[kDivModeIdx]->ToBool(),
                    (*outputTuple)[kY1OutIdx]->ToTensor(), (*outputTuple)[kY2OutIdx]->ToTensor(),
                    (*outputTuple)[kXOutIdx]->ToTensor(), nullptr);

  return SUCCESS;
}

MRT_REG_OP(add_rms_norm_quant, AclnnAddRmsNormQuantV2, Ascend);
}  // namespace ops
}  // namespace mrt
