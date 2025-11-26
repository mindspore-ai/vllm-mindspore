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

#include "ops/ascend/lowered/mlir_compiler.h"

#include <limits.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <sstream>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/SHA256.h"

#include "common/logger.h"
#include "hardware/ascend/res_manager/symbol_interface/symbol_utils.h"

namespace mrt::ops {

// ============================================================
// MlirCompiler Implementation
// ============================================================

MlirCompiler::MlirCompiler() {
  // Initialize options with defaults from environment variables
  InitializeDefaultOptions();

  // Create cache directory
  if (options_.enableCache) {
    std::string mkdirCmd = "mkdir -p " + options_.cacheDir;
    int ret = system(mkdirCmd.c_str());
    (void)ret;  // Ignore return value
  }
}

MlirCompiler &MlirCompiler::Instance() {
  static MlirCompiler instance;
  return instance;
}

void MlirCompiler::InitializeDefaultOptions() {
  // Read cache directory from environment variable
  const char *cacheDirEnv = std::getenv("INFERRT_MLIR_CACHE_DIR");
  if (cacheDirEnv != nullptr) {
    options_.cacheDir = cacheDirEnv;
  } else {
    // Use current directory as cache location
    char cwdBuffer[PATH_MAX];
    if (getcwd(cwdBuffer, sizeof(cwdBuffer)) != nullptr) {
      options_.cacheDir = std::string(cwdBuffer) + "/.mrt_lowered_cache";
    } else {
      options_.cacheDir = ".mrt_lowered_cache";  // Fallback
    }
  }

  // Read bishengir-compile path from environment variable
  const char *bishengirEnv = std::getenv("BISHENGIR_COMPILE");
  if (bishengirEnv != nullptr) {
    options_.bishengirCompilePath = bishengirEnv;
  } else {
    options_.bishengirCompilePath = "bishengir-compile";
  }

  // Read verbose flag from environment variable
  const char *verboseEnv = std::getenv("INFERRT_MLIR_VERBOSE");
  if (verboseEnv != nullptr && (std::string(verboseEnv) == "1" || std::string(verboseEnv) == "true")) {
    options_.verbose = true;
  } else {
    options_.verbose = false;
  }

  // Always enable cache by default
  options_.enableCache = true;

  LOG_OUT << "MlirCompiler initialized with options:";
  LOG_OUT << "  cacheDir: " << options_.cacheDir;
  LOG_OUT << "  bishengirCompilePath: " << options_.bishengirCompilePath;
  LOG_OUT << "  verbose: " << (options_.verbose ? "true" : "false");
  LOG_OUT << "  enableCache: " << (options_.enableCache ? "true" : "false");
}

void MlirCompiler::SetOptions(const CompileOptions &options) {
  std::lock_guard<std::mutex> lock(mutex_);
  options_ = options;

  // Create cache directory if needed
  if (options_.enableCache) {
    std::string mkdirCmd = "mkdir -p " + options_.cacheDir;
    int ret = system(mkdirCmd.c_str());
    (void)ret;
  }
}

std::string MlirCompiler::ComputeHash(const std::string &mlirContent) const {
  auto hash = llvm::SHA256::hash(llvm::ArrayRef<uint8_t>(
      reinterpret_cast<const uint8_t*>(mlirContent.data()), mlirContent.size()));

  std::ostringstream oss;
  for (auto byte : hash) {
    oss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(byte);
  }

  return oss.str();
}

bool MlirCompiler::CheckCache(const std::string &hash, CompileResult &result) {
  if (!options_.enableCache) {
    return false;
  }

  // Check in-memory cache first
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = cacheMap_.find(hash);
    if (it != cacheMap_.end() && it->second.success) {
      // Verify .so file still exists
      struct stat st;
      if (stat(it->second.soPath.c_str(), &st) == 0) {
        result = it->second;
        LOG_OUT << "MLIR cache hit (in-memory): " << hash;
        return true;
      }
    }
  }

  // Check disk cache
  std::string soPath = options_.cacheDir + "/" + hash + ".so";
  std::string metaPath = options_.cacheDir + "/" + hash + ".meta";

  struct stat st;
  if (stat(soPath.c_str(), &st) != 0) {
    return false;  // .so file doesn't exist
  }

  // Read metadata file
  std::ifstream metaFile(metaPath);
  if (!metaFile.is_open()) {
    LOG_OUT << "MLIR cache metadata not found: " << metaPath;
    return false;
  }

  std::string entryName, tilingPrefix;
  std::getline(metaFile, entryName);
  std::getline(metaFile, tilingPrefix);
  metaFile.close();

  if (entryName.empty()) {
    LOG_OUT << "Invalid MLIR cache metadata: " << metaPath;
    return false;
  }

  result.success = true;
  result.soPath = soPath;
  result.entryName = entryName;
  result.tilingPrefix = tilingPrefix;

  // Update in-memory cache
  {
    std::lock_guard<std::mutex> lock(mutex_);
    cacheMap_[hash] = result;
  }

  LOG_OUT << "MLIR cache hit (disk): " << hash;
  return true;
}

void MlirCompiler::SaveToCache(const std::string &hash, const CompileResult &result) {
  if (!options_.enableCache || !result.success) {
    return;
  }

  // Save to in-memory cache
  {
    std::lock_guard<std::mutex> lock(mutex_);
    cacheMap_[hash] = result;
  }

  // Save metadata to disk
  std::string metaPath = options_.cacheDir + "/" + hash + ".meta";
  std::ofstream metaFile(metaPath);
  if (!metaFile.is_open()) {
    LOG_ERROR << "Failed to write MLIR cache metadata: " << metaPath;
    return;
  }

  metaFile << result.entryName << "\n";
  metaFile << result.tilingPrefix << "\n";
  metaFile.close();

  LOG_OUT << "MLIR cache saved: " << hash;
}

bool MlirCompiler::ReadFileContent(const std::string &filePath, std::string &content) {
  std::ifstream file(filePath);
  if (!file.is_open()) {
    LOG_ERROR << "Failed to open file: " << filePath;
    return false;
  }

  std::ostringstream oss;
  oss << file.rdbuf();
  content = oss.str();
  file.close();

  return true;
}

bool MlirCompiler::WriteFileContent(const std::string &filePath, const std::string &content) {
  std::ofstream file(filePath);
  if (!file.is_open()) {
    LOG_ERROR << "Failed to write file: " << filePath;
    return false;
  }

  file << content;
  file.close();

  return true;
}

bool MlirCompiler::ExecuteCommand(const std::string &command, std::string &output, int &exitCode) {
  if (options_.verbose) {
    LOG_OUT << "Executing command: " << command;
  }

  // Open pipe to command
  FILE *pipe = popen((command + " 2>&1").c_str(), "r");
  if (pipe == nullptr) {
    LOG_ERROR << "Failed to execute command: " << command;
    exitCode = -1;
    return false;
  }

  // Read command output
  char buffer[256];
  std::ostringstream oss;
  while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
    oss << buffer;
  }

  output = oss.str();
  exitCode = pclose(pipe);

  if (options_.verbose) {
    LOG_OUT << "Command output:\n" << output;
    LOG_OUT << "Command exit code: " << exitCode;
  }

  return true;
}

bool MlirCompiler::RunBishengirCompile(const std::string &linalgFile, const std::string &outputSo) {
  // Get SoC name from ACL interface
  const char *socName = mrt::device::ascend::GetAscendSocVersion();
  if (socName == nullptr) {
    LOG_ERROR << "Failed to get SoC name from ACL";
    return false;
  }

  // Run bishengir-compile to generate .so from Linalg IR
  std::ostringstream cmd;
  cmd << options_.bishengirCompilePath << " " << linalgFile << " --enable-hfusion-compile=true"
      << " --enable-hivm-compile=true" << " --enable-auto-multi-buffer=true" << " --target=" << socName
      << " --enable-bin-relocation=false" << " -o " << outputSo;

  LOG_OUT << "Bishengir compile command: " << cmd.str();

  std::string output;
  int exitCode;
  if (!ExecuteCommand(cmd.str(), output, exitCode)) {
    LOG_ERROR << "Failed to execute bishengir-compile";
    return false;
  }

  if (exitCode != 0) {
    LOG_ERROR << "bishengir-compile failed with exit code " << exitCode << ":\n" << output;
    return false;
  }

  LOG_OUT << "bishengir-compile completed: " << outputSo;
  return true;
}

bool MlirCompiler::ExtractFunctionNames(const std::string &mlirText, std::string &entryName,
                                        std::string &tilingPrefix) {
  // Extract function name from MLIR text using manual parsing
  // Look for pattern: func.func @function_name
  const std::string pattern = "func.func @";
  size_t pos = mlirText.find(pattern);

  if (pos == std::string::npos) {
    entryName.clear();
    tilingPrefix.clear();
    LOG_ERROR << "Failed to find 'func.func @' pattern in MLIR";
    return false;
  }

  // Skip past "func.func @"
  pos += pattern.length();

  // Skip any whitespace after '@'
  while (pos < mlirText.size() && std::isspace(mlirText[pos])) {
    ++pos;
  }

  // Extract function name (alphanumeric and underscore characters)
  size_t start = pos;
  while (pos < mlirText.size() &&
         (std::isalnum(static_cast<unsigned char>(mlirText[pos])) || mlirText[pos] == '_')) {
    ++pos;
  }

  if (pos > start) {
    entryName = mlirText.substr(start, pos - start);
    tilingPrefix = entryName;
    LOG_OUT << "Extracted function name: " << entryName;
    return true;
  }

  entryName.clear();
  tilingPrefix.clear();
  LOG_ERROR << "Failed to extract function name from MLIR";
  return false;
}

MlirCompiler::CompileResult MlirCompiler::CompileFromText(const std::string &mlirText) {
  CompileResult result;

  if (mlirText.empty()) {
    result.errorMessage = "Empty MLIR text";
    LOG_ERROR << result.errorMessage;
    return result;
  }

  // Compute hash for caching
  std::string hash = ComputeHash(mlirText);

  // Check cache
  if (CheckCache(hash, result)) {
    return result;
  }

  LOG_OUT << "Compiling MLIR text (hash: " << hash << ")";

  // Extract function names from MLIR text
  std::string entryName, tilingPrefix;
  if (!ExtractFunctionNames(mlirText, entryName, tilingPrefix)) {
    result.errorMessage = "Failed to extract function name from MLIR";
    LOG_ERROR << result.errorMessage;
    return result;
  }

  // Create temporary files
  std::string inputFile = options_.cacheDir + "/" + hash + "_input.mlir";
  std::string outputSo = options_.cacheDir + "/" + hash + ".so";

  // Write MLIR text to file
  if (!WriteFileContent(inputFile, mlirText)) {
    result.errorMessage = "Failed to write MLIR text to file";
    LOG_ERROR << result.errorMessage;
    return result;
  }

  // Run bishengir-compile (input is assumed to be Linalg IR)
  if (!RunBishengirCompile(inputFile, outputSo)) {
    result.errorMessage = "bishengir-compile failed";
    LOG_ERROR << result.errorMessage;
    return result;
  }

  // bishengir-compile adds "lib" prefix to the output filename
  // If outputSo is "path/hash.so", actual file is "path/libhash.so"
  std::string actualSo;
  size_t lastSlash = outputSo.find_last_of('/');
  if (lastSlash != std::string::npos) {
    actualSo = outputSo.substr(0, lastSlash + 1) + "lib" + outputSo.substr(lastSlash + 1);
  } else {
    actualSo = "lib" + outputSo;
  }

  // Set function names
  result.entryName = entryName;
  result.tilingPrefix = tilingPrefix;
  LOG_OUT << "Entry function: " << result.entryName;
  LOG_OUT << "Tiling prefix: " << result.tilingPrefix;

  // Clean up temporary input file (keep .so)
  unlink(inputFile.c_str());

  // Fill result
  result.success = true;
  result.soPath = actualSo;  // Use the actual .so path with "lib" prefix

  // Save to cache
  SaveToCache(hash, result);

  LOG_OUT << "MLIR compilation successful: " << actualSo;
  return result;
}

void MlirCompiler::ClearCache() {
  std::lock_guard<std::mutex> lock(mutex_);

  cacheMap_.clear();

  // Remove all files in cache directory
  std::string cmd = "rm -rf " + options_.cacheDir + "/*";
  int ret = system(cmd.c_str());
  (void)ret;

  LOG_OUT << "MLIR cache cleared";
}

void MlirCompiler::GetCacheStats(size_t *totalEntries, size_t *cacheSizeBytes) const {
  std::lock_guard<std::mutex> lock(mutex_);

  if (totalEntries != nullptr) {
    *totalEntries = cacheMap_.size();
  }

  if (cacheSizeBytes != nullptr) {
    *cacheSizeBytes = 0;

    // Calculate total size of .so files
    for (const auto &entry : cacheMap_) {
      struct stat st;
      if (stat(entry.second.soPath.c_str(), &st) == 0) {
        *cacheSizeBytes += st.st_size;
      }
    }
  }
}

}  // namespace mrt::ops
