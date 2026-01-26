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
#include "ops/ascend/aclnn/inplace_copy.h"
#include <vector>
#include "ops/ascend/aclnn/utils/opapi_utils.h"
#include "ops/op_base/utils.h"
#include "ops/op_register.h"

namespace mrt {
namespace ops {
OpsErrorCode AclnnInplaceCopy::CalcWorkspace(const std::vector<const ir::Value *> &input, const ir::Value *output,
                                             size_t *workspaceSize) {
  auto dst = input[kIndex0]->ToTensor();
  CHECK_IF_FAIL(input[kIndex1]->IsTensor());
  auto src = input[kIndex1]->ToTensor();
  non_blocking_ = input[kIndex2]->ToBool();
  bool srcNpu = src->GetDevice().type == hardware::DeviceType::NPU;
  bool dstNpu = dst->GetDevice().type == hardware::DeviceType::NPU;

  if (dstNpu && srcNpu) {
    copyMode_ = mrt::device::CopyType::D2D;
    srcContiguous_ = src->IsContiguous();
    if (!srcContiguous_) {
      executor_->GetWorkspaceSize(static_cast<uint64_t *>(workspaceSize), dst, src);
    }
  } else if (dstNpu && !srcNpu) {
    copyMode_ = mrt::device::CopyType::H2D;
  } else if (!dstNpu && srcNpu) {
    copyMode_ = mrt::device::CopyType::D2H;
  } else if (!dstNpu && !srcNpu) {
    copyMode_ = mrt::device::CopyType::H2H;
  }

  if (copyMode_ != mrt::device::CopyType::D2D &&
      (dst->Dtype() != src->Dtype() || dst->Shape() != src->Shape() || !dst->IsContiguous() || !src->IsContiguous())) {
    LOG_EXCEPTION << "InplaceCopy H2D/D2H/H2H don't support BroadCast, DtypeCast, discontiguous src/dst yet.";
  }
  return SUCCESS;
}

OpsErrorCode AclnnInplaceCopy::Launch(const std::vector<const ir::Value *> &input, void *workspace,
                                      size_t workspaceSize, ir::Value *output, void *stream) {
  auto dst = input[kIndex0]->ToTensor();
  auto src = input[kIndex1]->ToTensor();

  if (copyMode_ == mrt::device::CopyType::D2D) {
    if (!srcContiguous_) {
      executor_->Launch(workspace, workspaceSize, stream, dst, src);
      return SUCCESS;
    }
    // For D2D copy, if src is contiguous, use direct async copy for better performance
    size_t srcSize = src->Numel() * static_cast<size_t>(dst->Dtype().GetSize());
    auto ret = res_manager_->AsyncCopy(dst->DataPtr(), src->DataPtr(), srcSize, copyMode_, stream);

    if (!ret) {
      LOG_ERROR << "Call aclrtMemcpyAsync in Op InplaceCopy failed";
      return UNKNOWN_ERROR;
    }
    return SUCCESS;
  }

  size_t srcSize = src->Numel() * static_cast<size_t>(dst->Dtype().GetSize());
  if (non_blocking_) {
    res_manager_->AsyncCopy(dst->DataPtr(), src->DataPtr(), srcSize, copyMode_, stream);
  } else {
    stream_mng_->SyncStream(stream);
    res_manager_->SyncCopy(dst->DataPtr(), src->DataPtr(), srcSize, copyMode_);
  }
  return SUCCESS;
}

MRT_REG_OP(inplace_copy, AclnnInplaceCopy, Ascend);
}  // namespace ops
}  // namespace mrt
