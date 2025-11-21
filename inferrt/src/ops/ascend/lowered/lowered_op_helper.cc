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

#include <fstream>
#include <sstream>
#include <functional>

#include "common/logger.h"
#include "ops/ascend/lowered/auto_lowered_op.h"
#include "ops/ascend/lowered/kernel_spec.h"

namespace mrt::ops {

// Generate kernel id from mlir_text hash
static std::string GenerateKernelId(const std::string &mlir_text) {
  std::size_t hash_value = std::hash<std::string>{}(mlir_text);
  return "kernel_" + std::to_string(hash_value);
}

std::unique_ptr<Operator> LoweredOpHelper::CreateFromMlirText(const std::string &mlir_text) {
  if (mlir_text.empty()) {
    LOG_ERROR << "MLIR text is empty";
    return nullptr;
  }

  std::string kernel_id = GenerateKernelId(mlir_text);

  if (KernelRegistry::Instance().Contains(kernel_id)) {
    LOG_OUT << "Kernel already registered: " << kernel_id << ", reusing";
    try {
      return std::make_unique<AutoLoweredOp>(kernel_id);
    } catch (const std::exception &e) {
      LOG_ERROR << "Failed to create AutoLoweredOp for " << kernel_id << ": " << e.what();
      return nullptr;
    }
  }

  if (!KernelRegistry::Instance().Register(kernel_id, mlir_text)) {
    LOG_ERROR << "Failed to register kernel spec: " << kernel_id;
    return nullptr;
  }

  LOG_OUT << "Registered lowered kernel: " << kernel_id;

  try {
    return std::make_unique<AutoLoweredOp>(kernel_id);
  } catch (const std::exception &e) {
    LOG_ERROR << "Failed to create AutoLoweredOp for " << kernel_id << ": " << e.what();
    return nullptr;
  }
}

}  // namespace mrt::ops
