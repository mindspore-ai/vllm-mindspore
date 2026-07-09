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

#include "ops/ascend/lowered/lowered_op_helper.h"

#include "common/logger.h"
#include "ops/ascend/lowered/auto_lowered_op.h"

namespace mrt::ops {

std::unique_ptr<Operator> LoweredOpHelper::CreateFromMlirText(const std::string &mlir_text) {
  if (mlir_text.empty()) {
    RT_GLOG(ERROR) << "MLIR text is empty";
    return nullptr;
  }

  try {
    return std::make_unique<AutoLoweredOp>(mlir_text);
  } catch (const std::exception &e) {
    RT_GLOG(ERROR) << "Failed to create AutoLoweredOp: " << e.what();
    return nullptr;
  }
}

}  // namespace mrt::ops
