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
#include <mutex>

#include "common/visible.h"

namespace mrt::ops {

/**
 * @brief MLIR Compiler for Linalg → BiSheng IR → .so
 *
 * This class provides the compilation pipeline:
 * 1. Linalg → BiSheng IR (via bishengir-compile)
 * 2. BiSheng IR → .so file (via backend compiler)
 *
 * Features:
 * - Compilation caching based on MLIR content hash
 * - Thread-safe compilation
 * - Configurable output directory
 * - Automatic entry/tiling function name extraction
 */
class MlirCompiler {
 public:
  /**
   * @brief Compilation options
   */
  struct CompileOptions {
    std::string cacheDir =
      ".mrt_lowered_cache";  // Cache directory for compiled kernels (default: current_dir/.mrt_lowered_cache)
    std::string bishengirCompilePath = "bishengir-compile";  // Path to bishengir-compile tool
    bool enableCache = true;                                  // Enable compilation cache
    bool verbose = false;                                      // Enable verbose logging

    CompileOptions() = default;
  };

  /**
   * @brief Compilation result
   */
  struct CompileResult {
    bool success = false;       // Compilation success flag
    std::string soPath;        // Output .so file path
    std::string entryName;     // Host API entry function name
    std::string tilingPrefix;  // Tiling function prefix
    std::string errorMessage;  // Error message if compilation failed

    CompileResult() = default;
  };

  /**
   * @brief Get the singleton instance
   * @return Reference to the global MlirCompiler instance
   */
  static MlirCompiler &Instance();

  /**
   * @brief Set compilation options
   * @param options Compilation options to set
   */
  void SetOptions(const CompileOptions &options);

  /**
   * @brief Get current compilation options
   * @return Current compilation options
   */
  const CompileOptions &GetOptions() const { return options_; }

  /**
   * @brief Compile MLIR text to .so file
   *
   * Pipeline:
   * 1. Hash the MLIR text for cache lookup
   * 2. Check if cached .so exists and is valid
   * 3. If not cached:
   *    a. Write MLIR text to temporary file
   *    b. Run bishengir-compile to generate .so
   *    c. Extract entry/tiling function names
   *    d. Cache the result
   * 4. Return compilation result
   *
   * @param mlirText MLIR text (Linalg dialect)
   * @return Compilation result with .so path and entry names
   */
  CompileResult CompileFromText(const std::string &mlirText);
  /**
   * @brief Clear compilation cache
   * Removes all cached .so files and metadata
   */
  void ClearCache();

  /**
   * @brief Get cache statistics
   * @param totalEntries Output: total number of cached kernels
   * @param cacheSizeBytes Output: total cache size in bytes
   */
  void GetCacheStats(size_t *totalEntries, size_t *cacheSizeBytes) const;

  // Disable copy and move
  MlirCompiler(const MlirCompiler &) = delete;
  MlirCompiler &operator=(const MlirCompiler &) = delete;
  MlirCompiler(MlirCompiler &&) = delete;
  MlirCompiler &operator=(MlirCompiler &&) = delete;

 private:
  MlirCompiler();
  ~MlirCompiler() = default;

  /**
   * @brief Initialize default options from environment variables
   *
   * Reads configuration from:
   * - INFERRT_MLIR_CACHE_DIR: Cache directory (default: <current_dir>/.mrt_lowered_cache)
   * - INFERRT_MLIR_VERBOSE: Enable verbose logging (default: false)
   * - BISHENGIR_COMPILE: Path to bishengir-compile tool (default: bishengir-compile)
   */
  void InitializeDefaultOptions();

  /**
   * @brief Compute hash of MLIR content
   * @param mlirContent MLIR text
   * @return Hash string (hex)
   */
  std::string ComputeHash(const std::string &mlirContent) const;

  /**
   * @brief Check if cached .so exists and is valid
   * @param hash MLIR content hash
   * @param result Output: cached compilation result
   * @return true if cache hit and valid
   */
  bool CheckCache(const std::string &hash, CompileResult &result);

  /**
   * @brief Save compilation result to cache
   * @param hash MLIR content hash
   * @param result Compilation result to cache
   */
  void SaveToCache(const std::string &hash, const CompileResult &result);

  /**
   * @brief Run bishengir-compile to generate .so
   * @param linalgFile Input Linalg MLIR file
   * @param outputSo Output .so file path
   * @return true on success
   */
  bool RunBishengirCompile(const std::string &linalgFile, const std::string &outputSo);

  /**
   * @brief Extract entry and tiling function names from MLIR text
   * @param mlirText MLIR text content
   * @param entryName Output: entry function name
   * @param tilingPrefix Output: tiling function prefix
   * @return true on success
   */
  bool ExtractFunctionNames(const std::string &mlirText, std::string &entryName, std::string &tilingPrefix);

  /**
   * @brief Read file content to string
   * @param filePath File path
   * @param content Output: file content
   * @return true on success
   */
  bool ReadFileContent(const std::string &filePath, std::string &content);

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
  bool ExecuteCommand(const std::string &command, std::string &output, int &exitCode);

  CompileOptions options_;
  mutable std::mutex mutex_;
  std::unordered_map<std::string, CompileResult> cacheMap_;  // In-memory cache
};

}  // namespace mrt::ops

#endif  // __OPS_ASCEND_LOWERED_MLIR_COMPILER_H__
