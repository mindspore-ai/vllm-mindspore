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

#include "ops/cpu/symbolic/broadcast_shape.h"

#include <algorithm>
#include <vector>

#include "common/logger.h"
#include "ir/value/value.h"
#include "ops/op_register.h"

namespace mrt {
namespace ops {

namespace {
std::vector<int64_t> BroadcastShapes(const std::vector<int64_t> &a, const std::vector<int64_t> &b) {
  const size_t ra = a.size();
  const size_t rb = b.size();
  const size_t r = std::max(ra, rb);

  std::vector<int64_t> aa(r, 1);
  std::vector<int64_t> bb(r, 1);
  std::copy(a.begin(), a.end(), aa.begin() + (r - ra));
  std::copy(b.begin(), b.end(), bb.begin() + (r - rb));

  std::vector<int64_t> out;
  out.reserve(r);
  for (size_t i = 0; i < r; ++i) {
    int64_t da = aa[i];
    int64_t db = bb[i];
    if (da == db) {
      out.push_back(da);
      continue;
    }
    if (da == 1) {
      out.push_back(db);
      continue;
    }
    if (db == 1) {
      out.push_back(da);
      continue;
    }
    // Conservative handling for unknown dims: propagate unknown when any side is unknown.
    if (da < 0 || db < 0) {
      out.push_back(-1);
      continue;
    }
    LOG_EXCEPTION << "BroadcastShape: incompatible shapes for broadcasting: " << ir::ShapeToString(a) << " and "
                  << ir::ShapeToString(b);
  }
  return out;
}
}  // namespace

OpsErrorCode BroadcastShape::InferShape(const std::vector<const ir::Value *> &input, ir::Value *output) {
  if (input.size() != kInputSize2) {
    LOG_ERROR << "BroadcastShape::InferShape expects 2 inputs, but got: " << input.size();
    return INVALID_INPUT_NUM;
  }
  if (!input[kIndex0] || !input[kIndex0]->IsTensor() || !input[kIndex1] || !input[kIndex1]->IsTensor()) {
    LOG_ERROR << "BroadcastShape::InferShape expects 2 tensor inputs";
    return INVALID_PARAM;
  }

  auto t0 = input[kIndex0]->ToTensor();
  auto t1 = input[kIndex1]->ToTensor();

  // Don't call EvalSymbolicShape() here - it modifies the input tensors and may trigger
  // resize on already-allocated device memory, causing memory leak detection.
  // Instead, directly use the current shape. If symbolic shapes are present but not yet
  // evaluated, the shape will contain -1 for dynamic dims, which is acceptable for
  // InferShape purposes.

  const auto &s0 = t0->Shape();
  const auto &s1 = t1->Shape();

  auto outShape = BroadcastShapes(s0, s1);
  std::vector<ir::ValuePtr> dims;
  dims.reserve(outShape.size());
  std::transform(outShape.begin(), outShape.end(), std::back_inserter(dims),
                 [](int64_t d) { return ir::MakeIntrusive<ir::Value>(static_cast<int64_t>(d)); });
  *output = ir::Value(ir::MakeIntrusive<ir::Tuple>(std::move(dims)));
  return SUCCESS;
}

OpsErrorCode BroadcastShape::Launch(const std::vector<const ir::Value *> &input, void *workspace, size_t workspaceSize,
                                    ir::Value *output, void *stream) {
  (void)input;
  (void)workspace;
  (void)workspaceSize;
  (void)output;
  (void)stream;
  // Host-side metadata op: no device launch needed.
  return SUCCESS;
}

MRT_REG_OP(broadcast_shape, BroadcastShape, CPU);

}  // namespace ops
}  // namespace mrt
