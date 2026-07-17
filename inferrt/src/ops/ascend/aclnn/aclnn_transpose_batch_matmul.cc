/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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

#include "ops/ascend/aclnn/aclnn_transpose_batch_matmul.h"

#include <algorithm>
#include <cstdint>
#include <vector>

#include "ir/common/dtype.h"
#include "ops/ascend/aclnn/utils/opapi_utils.h"
#include "ops/op_register.h"

namespace mrt {
namespace ops {
namespace {
constexpr size_t kInputIdx = 0;
constexpr size_t kWeightIdx = 1;
constexpr size_t kBiasIdx = 2;
constexpr size_t kScaleIdx = 3;
constexpr size_t kPermX1Idx = 4;
constexpr size_t kPermX2Idx = 5;
constexpr size_t kPermYIdx = 6;
constexpr size_t kBatchSplitFactorIdx = 7;
constexpr size_t kExpectedInputNum = 8;
constexpr size_t kExpectedRank = 3;

const std::vector<int64_t> kDefaultPermX = {0, 1, 2};
const std::vector<int64_t> kDefaultPermY = {1, 0, 2};

bool IsSupportedInputDtype(const ir::DataType &dtype) {
  return dtype == ir::DataType::Float16 || dtype == ir::DataType::Float32 || dtype == ir::DataType::BFloat16;
}

std::optional<ir::TensorPtr> GetOptionalTensor(const ir::Value *value) {
  return value->IsTensor() ? std::optional(value->ToTensor()) : std::nullopt;
}

std::vector<int64_t> GetIntListOrDefault(const ir::Value *value, const std::vector<int64_t> &defaultValue) {
  return value->IsTuple() ? value->ToTuple()->ToIntList() : defaultValue;
}

bool IsSamePerm(const std::vector<int64_t> &perm, const std::vector<int64_t> &expected) {
  return perm.size() == expected.size() && std::equal(perm.begin(), perm.end(), expected.begin());
}

void CheckInputRank(const ir::TensorPtr &tensor, const char *name) {
  CHECK_IF_NULL(tensor);
  if (tensor->Dim() != kExpectedRank) {
    RT_GLOG(EXCEPTION) << "npu_transpose_batchmatmul: " << name << " dim is " << tensor->Dim() << ", but expected is "
                       << kExpectedRank;
  }
}

void CheckInputDtype(const ir::TensorPtr &tensor, const char *name) {
  CHECK_IF_NULL(tensor);
  if (!IsSupportedInputDtype(tensor->Dtype())) {
    RT_GLOG(EXCEPTION) << "npu_transpose_batchmatmul: " << name
                       << "'s type supported for float16, float32 and bfloat16, but got " << tensor->Dtype().ToString();
  }
}
}  // namespace

void AclnnTransposeBatchMatmul::ParseAndValidateInputs(const std::vector<const ir::Value *> &input) {
  if (input.size() != kExpectedInputNum) {
    RT_GLOG(EXCEPTION) << "npu_transpose_batchmatmul expects " << kExpectedInputNum << " inputs, but got "
                       << input.size();
  }

  input_ = input[kInputIdx]->ToTensor();
  weight_ = input[kWeightIdx]->ToTensor();
  bias_ = GetOptionalTensor(input[kBiasIdx]);
  scale_ = GetOptionalTensor(input[kScaleIdx]);
  permX1_ = GetIntListOrDefault(input[kPermX1Idx], kDefaultPermX);
  permX2_ = GetIntListOrDefault(input[kPermX2Idx], kDefaultPermX);
  permY_ = GetIntListOrDefault(input[kPermYIdx], kDefaultPermY);
  batchSplitFactor_ =
    input[kBatchSplitFactorIdx]->IsNone() ? 1 : static_cast<int32_t>(input[kBatchSplitFactorIdx]->ToInt());
  cubeMathType_ = GetCubeMathType();

  CheckInputRank(input_, "input");
  CheckInputRank(weight_, "weight");
  CheckInputDtype(input_, "input");
  CheckInputDtype(weight_, "weight");

  const auto checkPermX1 = IsSamePerm(permX1_, kDefaultPermX) || IsSamePerm(permX1_, kDefaultPermY);
  if (!checkPermX1) {
    RT_GLOG(EXCEPTION) << "npu_transpose_batchmatmul: perm_x1 should be [0, 1, 2] or [1, 0, 2], but got " << permX1_;
  }
  if (!IsSamePerm(permX2_, kDefaultPermX)) {
    RT_GLOG(EXCEPTION) << "npu_transpose_batchmatmul: perm_x2 should be [0, 1, 2], but got " << permX2_;
  }
  if (!IsSamePerm(permY_, kDefaultPermY)) {
    RT_GLOG(EXCEPTION) << "npu_transpose_batchmatmul: perm_y should be [1, 0, 2], but got " << permY_;
  }
  if (bias_.has_value()) {
    RT_GLOG(EXCEPTION) << "npu_transpose_batchmatmul: bias is not supported";
  }
}

OpsErrorCode AclnnTransposeBatchMatmul::CalcWorkspace(const std::vector<const ir::Value *> &input,
                                                      const ir::Value *output, size_t *workspaceSize) {
  ParseAndValidateInputs(input);
  executor_->GetWorkspaceSize(static_cast<uint64_t *>(workspaceSize), input_, weight_, bias_, scale_, permX1_, permX2_,
                              permY_, cubeMathType_, batchSplitFactor_, output->ToTensor());
  return SUCCESS;
}

OpsErrorCode AclnnTransposeBatchMatmul::Launch(const std::vector<const ir::Value *> &input, void *workspace,
                                               size_t workspaceSize, ir::Value *output, void *stream) {
  executor_->Launch(workspace, workspaceSize, stream, input_, weight_, bias_, scale_, permX1_, permX2_, permY_,
                    cubeMathType_, batchSplitFactor_, output->ToTensor());
  return SUCCESS;
}

MRT_REG_OP(npu_transpose_batchmatmul, AclnnTransposeBatchMatmul, Ascend);
}  // namespace ops
}  // namespace mrt
