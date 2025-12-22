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

#include "ops/ascend/aclnn/expand.h"

#include <algorithm>
#include <limits>
#include <vector>

#include "common/logger.h"
#include "ir/value/value.h"
#include "ops/op_register.h"

namespace mrt {
namespace ops {

AclnnExpand::AclnnExpand() { executor_ = std::make_unique<AclnnExecutor>("aclnnExpand"); }

OpsErrorCode AclnnExpand::InferShape(const std::vector<const ir::Value *> &input, ir::Value *output) {
  // Runtime executes InferShape() before CalcWorkspace()/Launch in dynamic-shape mode.
  // For expand, output rank/shape is determined by the `size` operand, and `-1`
  // means "keep the corresponding input dimension" (PyTorch expand semantics).
  if (input.size() < 2) {
    LOG_ERROR << "AclnnExpand::InferShape expects at least 2 inputs, but got: " << input.size();
    return INVALID_INPUT_NUM;
  }
  if (!input[0] || !input[0]->IsTensor()) {
    LOG_ERROR << "AclnnExpand::InferShape expects input[0] to be a tensor";
    return INVALID_PARAM;
  }
  if (!input[1] || !input[1]->IsTuple()) {
    LOG_ERROR << "AclnnExpand::InferShape expects input[1] (size) to be a tuple<int>";
    return INVALID_PARAM;
  }
  if (!output || !output->IsTensor()) {
    LOG_ERROR << "AclnnExpand::InferShape expects output to be a tensor";
    return INVALID_PARAM;
  }

  const auto &inTensor = input[0]->ToTensor();
  // If upstream attached symbolic shape metadata, make sure it is evaluated
  // before consulting `inTensor->Shape()`.
  if (inTensor->HasSymbolicShape() && inTensor->HasDynamicShape()) {
    inTensor->EvalSymbolicShape();
  }
  const auto &inShape = inTensor->Shape();
  const auto &size = input[1]->ToTuple()->ToIntList();
  if (size.empty()) {
    LOG_ERROR << "AclnnExpand::InferShape got empty size";
    return INVALID_PARAM;
  }

  LOG_OUT << "AclnnExpand::InferShape: inShape=" << ir::ShapeToString(inShape) << ", size=" << ir::ShapeToString(size);

  // Align input shape to output rank (expand can prepend leading dims).
  if (inShape.size() > size.size()) {
    LOG_EXCEPTION << "AclnnExpand::InferShape: input rank " << inShape.size() << " > target rank " << size.size();
  }
  std::vector<int64_t> alignedIn(size.size(), 1);
  std::copy(inShape.begin(), inShape.end(), alignedIn.begin() + (size.size() - inShape.size()));

  std::vector<int64_t> outShape;
  outShape.reserve(size.size());
  for (size_t i = 0; i < size.size(); ++i) {
    int64_t target = size[i];
    int64_t inDim = alignedIn[i];
    // In our MLIR lowering, dynamic extents can appear as:
    // - -1 (ShapedType::kDynamic)
    // - INT64_MIN (ShapedType::kDynamicStrideOrOffset), which may leak into
    //   size lists in some paths. Treat both as "keep input dim".
    if (target == -1 || target == std::numeric_limits<int64_t>::min()) {
      target = inDim;
    }
    // Broadcast legality: inDim must be 1 or equal to target.
    if (inDim != target && inDim != 1) {
      LOG_EXCEPTION << "AclnnExpand::InferShape: cannot expand dim " << i << " from " << inDim << " to " << target;
    }
    outShape.push_back(target);
  }

  auto &outTensor = output->ToTensor();
  outTensor->SetShape(std::move(outShape));
  outTensor->Resize();  // updates numel_ and storage sizeBytes_
  return SUCCESS;
}

OpsErrorCode AclnnExpand::CalcWorkspace(const std::vector<const ir::Value *> &input, const ir::Value *output,
                                        size_t *workspaceSize) {
  LOG_OUT << "Begin CalcWorkspace for op [expand]";
  executor_->GetWorkspaceSize(static_cast<uint64_t *>(workspaceSize), input[0]->ToTensor(),
                              input[1]->ToTuple()->ToIntList(), output->ToTensor());
  return SUCCESS;
}

OpsErrorCode AclnnExpand::Launch(const std::vector<const ir::Value *> &input, void *workspace, size_t workspaceSize,
                                 ir::Value *output, void *stream) {
  LOG_OUT << "Begin Launch for op [expand]";
  executor_->Launch(workspace, workspaceSize, stream, input[0]->ToTensor(), input[1]->ToTuple()->ToIntList(),
                    output->ToTensor());
  return SUCCESS;
}

MRT_REG_OP(expand, AclnnExpand, Ascend);
}  // namespace ops
}  // namespace mrt
