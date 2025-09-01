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

#include "ops/op_base/utils.h"

namespace mrt {
namespace ops {
void CalBroadCastShape(const std::vector<int64_t> &xShape, const std::vector<int64_t> &yShape,
                       std::vector<int64_t> *broadcastShape) {
  if (xShape == yShape) {
    *broadcastShape = xShape;
    return;
  }

  auto xLength = xShape.size();
  auto yLength = yShape.size();
  auto res = xLength > yLength;
  size_t maxLen = res ? xLength : yLength;
  size_t minLen = res ? yLength : xLength;
  const std::vector<int64_t> &maxShape = res ? xShape : yShape;
  const std::vector<int64_t> &minShape = res ? yShape : xShape;

  *broadcastShape = maxShape;
  auto lengthDiff = maxLen - minLen;
  for (size_t i = 0; i < minLen; ++i) {
    auto dsti = lengthDiff + i;
    if (maxShape[dsti] == 1) {
      (*broadcastShape)[dsti] = minShape[i];
    } else if (maxShape[dsti] != minShape[i] && minShape[i] != 1) {
      LOG_EXCEPTION << "xShape[" << xLength + i << "] or yShape[" << yLength + i << "] must be when they are not equal"
                    << ", but got xShape=" << ir::ShapeToString(xShape) << ", yShape=" << ir::ShapeToString(yShape);
    }
  }
}

}  // namespace ops
}  // namespace mrt
