/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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

/**
 * Custom Operator Wrapper for MLIR Lowered BiasAdd
 *
 * Uses simplified LoweredOpHelper API - no internal implementation details exposed.
 * The operator dynamically compiles MLIR to .so on first execution.
 */

#include <fstream>
#include <sstream>
#include <cstdlib>
#include <memory>

#include "ops/operator.h"
#include "ops/custom_op_register.h"
#include "ops/ascend/lowered/lowered_op_helper.h"  // Only need this single header!
#include "common/logger.h"

namespace mrt {
namespace ops {

// Helper function to read MLIR file
static std::string ReadMlirFile(const std::string &file_path) {
  std::ifstream file(file_path);
  if (!file.is_open()) {
    LOG_ERROR << "Cannot open MLIR file: " << file_path;
    return "";
  }

  std::ostringstream oss;
  oss << file.rdbuf();
  return oss.str();
}

class LoweredBiasAddOperator : public Operator {
 public:
  LoweredBiasAddOperator() {
    const char *mlir_path_env = std::getenv("LOWERED_BIAS_ADD_MLIR_PATH");
    std::string mlir_path;

    if (mlir_path_env != nullptr) {
      mlir_path = mlir_path_env;
    } else {
      const char *test_dir = std::getenv("TEST_DIR");
      if (test_dir != nullptr) {
        mlir_path = std::string(test_dir) + "/bias_add_dyn.mlir";
      } else {
        mlir_path = "bias_add_dyn.mlir";
      }
    }

    std::string mlir_text = ReadMlirFile(mlir_path);
    if (mlir_text.empty()) {
      LOG_EXCEPTION << "Failed to read MLIR file: " << mlir_path
                    << ". Set LOWERED_BIAS_ADD_MLIR_PATH or TEST_DIR environment variable.";
    }

    lowered_op_ = LoweredOpHelper::CreateFromMlirText(mlir_text);

    if (lowered_op_ == nullptr) {
      LOG_EXCEPTION << "Failed to create lowered operator from MLIR";
    }

    LOG_OUT << "LoweredBiasAddOperator created successfully";
  }

  ~LoweredBiasAddOperator() override = default;

  OpsErrorCode InferShape(const std::vector<const ir::Value *> &input, ir::Value *output) override {
    if (lowered_op_ == nullptr) {
      LOG_ERROR << "Lowered operator not initialized";
      return UNKNOWN_ERROR;
    }
    return lowered_op_->InferShape(input, output);
  }

  OpsErrorCode CalcWorkspace(const std::vector<const ir::Value *> &input, const ir::Value *output,
                             size_t *workspaceSize) override {
    if (lowered_op_ == nullptr) {
      LOG_ERROR << "Lowered operator not initialized";
      return UNKNOWN_ERROR;
    }
    return lowered_op_->CalcWorkspace(input, output, workspaceSize);
  }

  OpsErrorCode Launch(const std::vector<const ir::Value *> &input, void *workspace, size_t workspaceSize,
                      ir::Value *output, void *stream) override {
    if (lowered_op_ == nullptr) {
      LOG_ERROR << "Lowered operator not initialized";
      return UNKNOWN_ERROR;
    }
    return lowered_op_->Launch(input, workspace, workspaceSize, output, stream);
  }

 private:
  std::unique_ptr<Operator> lowered_op_;
};

// Register custom operator
REGISTER_CUSTOM_OP(lowered_bias_add, LoweredBiasAddOperator);

}  // namespace ops
}  // namespace mrt
