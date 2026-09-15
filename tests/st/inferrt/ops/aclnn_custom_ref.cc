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

#include "include/custom_op_api.h"

namespace mrt {
namespace ops {

// ---------------------------------------------------------------------------
// In-place (ref) custom operator.
//   custom_inplace_add(self, other, alpha) -> self, computed as self += other * alpha.
// The output aliases input 0, so no memory is allocated for the output and the caller's
// tensor is updated in place.
// ---------------------------------------------------------------------------
class CustomInplaceAddOperator : public AclnnCustomOperator {
 public:
  // Defaults to ref pair {0, 0}: output 0 shares storage with input 0.
  CustomInplaceAddOperator() : AclnnCustomOperator("aclnnInplaceAdd") {}
  ~CustomInplaceAddOperator() override = default;

  OpsErrorCode CalcWorkspace(const std::vector<const ir::Value *> &input, const ir::Value *output,
                             size_t *workspaceSize) override {
    // aclnnInplaceAdd(selfRef, other, alpha)
    GetExecutor()->GetWorkspaceSize(static_cast<uint64_t *>(workspaceSize), input[kIndex0]->ToTensor(),
                                    input[kIndex1]->ToTensor(), input[kIndex2]);
    return SUCCESS;
  }

  OpsErrorCode Launch(const std::vector<const ir::Value *> &input, void *workspace, size_t workspaceSize,
                      ir::Value *output, void *stream) override {
    GetExecutor()->Launch(workspace, workspaceSize, stream, input[kIndex0]->ToTensor(), input[kIndex1]->ToTensor(),
                          input[kIndex2]);
    return SUCCESS;
  }

  std::vector<std::pair<uint32_t, uint32_t>> GetOutputInputRefPairs() const override { return {{0, 0}}; }
};

REGISTER_CUSTOM_OP(custom_inplace_add, CustomInplaceAddOperator);

// ---------------------------------------------------------------------------
// View (ref) custom operator.
//   custom_narrow_view(self, start, length) -> self[start : start + length] along dim 0.
// The output shares the storage of input 0 and only gets new shape / strides / storageOffset,
// so this is a zero-copy view: no kernel is launched and no memory is allocated.
// ---------------------------------------------------------------------------
class CustomNarrowViewOperator : public AclnnCustomViewOperator {
 public:
  CustomNarrowViewOperator() = default;
  ~CustomNarrowViewOperator() override = default;

  OpsErrorCode CalcWorkspace(const std::vector<const ir::Value *> &input, const ir::Value *output,
                             size_t *workspaceSize) override {
    const auto inputTensor = input[kIndex0]->ToTensor();
    const auto outputTensor = output->ToTensor();
    const auto start = input[kIndex1]->ToInt();

    CHECK_IF_FAIL_MSG(!outputTensor->HasDynamicShape(),
                      "custom_narrow_view output shape should have been inferred before CalcWorkspace");
    CHECK_IF_FAIL_MSG(!inputTensor->Shape().empty(), "custom_narrow_view can not be applied to a 0-dim tensor.");

    // Narrowing along dim 0 keeps the strides unchanged; only the storage offset moves.
    const auto strides = GetTensorStrides(inputTensor);
    const size_t newStorageOffset = static_cast<size_t>(inputTensor->StorageOffset() + start * strides[kIndex0]);
    UpdateTensorViewInfo(inputTensor, outputTensor, outputTensor->Shape(), strides, newStorageOffset);

    CheckStorageMatch(input, output);
    return SUCCESS;
  }
};

REGISTER_CUSTOM_OP(custom_narrow_view, CustomNarrowViewOperator);

}  // namespace ops
}  // namespace mrt
