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

#ifndef __OPS_ASCEND_LOWERED_MLIR_COMPILER_H__
#define __OPS_ASCEND_LOWERED_MLIR_COMPILER_H__

#include <string>
#include <memory>
#include <unordered_map>

#include "common/visible.h"

namespace mrt::ops {

/**
 * @brief MLIR Compiler for Linalg → BiSheng IR → .so
 *
 * Stateless compiler providing pure function-based compilation pipeline:
 * 1. Linalg → BiSheng IR (via bishengir-opt + bishengir-compile)
 * 2. BiSheng IR → .so file (via backend compiler)
 *
 * Features:
 * - Stateless: no global state, all config passed per-call
 * - Thread-safe: pure functions without side effects
 * - Caller-controlled paths and options
 */
namespace MlirCompiler {

/**
 * @brief Compilation request configuration
 */
struct CompileRequest {
  std::string mlirText;                                    // MLIR text to compile
  std::string cacheDir = ".mrt_lowered_cache";             // Directory for compiled kernels
  std::string bishengirCompilePath = "bishengir-compile";  // Path to bishengir-compile tool
  std::string bishengirOptPath = "bishengir-opt";          // Path to bishengir-opt tool
  bool keepIntermediateFiles = false;                      // Keep intermediate MLIR files for debugging
};

/**
 * @brief Compilation result
 */
struct CompileResult {
  bool success = false;      // Compilation success flag
  std::string cacheDir;      // Cache directory for this compilation (unique per compilation)
  std::string soPath;        // Output .so file path
  std::string entryName;     // Host API entry function name (also used as tiling function prefix)
  std::string errorMessage;  // Error message if compilation failed

  CompileResult() = default;
};

/**
 * @brief Compile MLIR text to .so file (stateless)
 *
 * Pipeline:
 * 1. Generate unique filename based on MLIR hash
 * 2. Write MLIR text to temporary file
 * 3. Run bishengir-opt and bishengir-compile to generate .so
 * 4. Extract entry/tiling function names
 * 5. Return compilation result
 *
 * @param request Compilation request with MLIR text and configuration
 * @return Compilation result with .so path and entry names
 */
CompileResult CompileFromText(const CompileRequest &request);

// ============================================================
// Internal helper functions
// ============================================================

namespace internal {

/**
 * @brief Initialize default compilation request from environment variables
 *
 * Reads configuration from:
 * - MRT_LOWERED_CACHE_DIR: Cache directory (default: <current_dir>/.mrt_lowered_cache)
 * - MRT_LOWERED_MLIR_KEEP_FILES: Keep intermediate files (default: false)
 * - BISHENGIR_COMPILE: Path to bishengir-compile tool (default: bishengir-compile)
 * - BISHENGIR_OPT: Path to bishengir-opt tool (default: bishengir-opt)
 */
CompileRequest InitializeDefaultRequest();

/**
 * @brief Compute hash of MLIR content
 * @param mlirContent MLIR text
 * @return Hash string (hex, first 16 chars)
 */
std::string ComputeHash(const std::string &mlirContent);

/**
 * @brief Run bishengir-compile to generate .so
 * @param linalgFile Input Linalg MLIR file
 * @param outputSo Output .so file path
 * @param bishengirCompilePath Path to bishengir-compile tool
 * @return true on success
 */
bool RunBishengirCompile(const std::string &linalgFile, const std::string &outputSo,
                         const std::string &bishengirCompilePath);

/**
 * @brief Run bishengir-opt to convert linalg.generic to named operations
 * @param inputFile Input MLIR file with linalg.generic ops
 * @param outputFile Output MLIR file with named ops
 * @param bishengirOptPath Path to bishengir-opt tool
 * @return true on success
 */
bool RunBishengirOpt(const std::string &inputFile, const std::string &outputFile, const std::string &bishengirOptPath);

/**
 * @brief Extract entry and tiling function names from MLIR text
 * @param mlirText MLIR text content
 * @param entryName Output: entry function name (also used as tiling prefix)
 * @return true on success
 */
bool ExtractFunctionNames(const std::string &mlirText, std::string *entryName);

/**
 * @brief Read file content to string
 * @param filePath File path
 * @param content Output: file content
 * @return true on success
 */
bool ReadFileContent(const std::string &filePath, std::string *content);

/**
 * @brief Write string content to file
 * @param filePath File path
 * @param content Content to write
 * @return true on success
 */
bool WriteFileContent(const std::string &filePath, const std::string &content);

/**
 * @brief Execute shell command and capture output
 * @param command Command to execute
 * @param output Output: command stdout/stderr
 * @param exitCode Output: command exit code
 * @return true if command executed (regardless of exit code)
 */
bool ExecuteCommand(const std::string &command, std::string *output, int *exitCode);

}  // namespace internal

}  // namespace MlirCompiler

}  // namespace mrt::ops

#endif  // __OPS_ASCEND_LOWERED_MLIR_COMPILER_H__
