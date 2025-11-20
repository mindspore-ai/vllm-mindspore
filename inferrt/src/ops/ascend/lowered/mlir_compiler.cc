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
// MlirCompiler Namespace Implementation
// ============================================================

namespace MlirCompiler {

namespace internal {

CompileRequest InitializeDefaultRequest() {
  CompileRequest request;

  // Read cache directory from environment variable
  const char *cacheDirEnv = std::getenv("MRT_LOWERED_CACHE_DIR");
  if (cacheDirEnv != nullptr) {
    request.cacheDir = cacheDirEnv;
  } else {
    // Use current directory as cache location
    char cwdBuffer[PATH_MAX];
    if (getcwd(cwdBuffer, sizeof(cwdBuffer)) != nullptr) {
      request.cacheDir = std::string(cwdBuffer) + "/.mrt_lowered_cache";
    } else {
      request.cacheDir = ".mrt_lowered_cache";  // Fallback
    }
  }

  // Read bishengir-compile path from environment variable
  const char *bishengirEnv = std::getenv("BISHENGIR_COMPILE");
  if (bishengirEnv != nullptr) {
    request.bishengirCompilePath = bishengirEnv;
  } else {
    request.bishengirCompilePath = "bishengir-compile";
  }

  // Read bishengir-opt path from environment variable
  const char *bishengirOptEnv = std::getenv("BISHENGIR_OPT");
  if (bishengirOptEnv != nullptr) {
    request.bishengirOptPath = bishengirOptEnv;
  } else {
    request.bishengirOptPath = "bishengir-opt";
  }

  // Read keep intermediate files flag from environment variable
  const char *keepFilesEnv = std::getenv("MRT_LOWERED_MLIR_KEEP_FILES");
  if (keepFilesEnv != nullptr && (std::string(keepFilesEnv) == "1" || std::string(keepFilesEnv) == "true")) {
    request.keepIntermediateFiles = true;
  }

  return request;
}

std::string ComputeHash(const std::string &mlirContent) {
  auto hash = llvm::SHA256::hash(
    llvm::ArrayRef<uint8_t>(reinterpret_cast<const uint8_t *>(mlirContent.data()), mlirContent.size()));

  std::ostringstream oss;
  // Only use first 16 chars for shorter filenames
  for (size_t i = 0; i < 8 && i < hash.size(); ++i) {
    oss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(hash[i]);
  }

  return oss.str();
}

bool ReadFileContent(const std::string &filePath, std::string *content) {
  std::ifstream file(filePath);
  if (!file.is_open()) {
    LOG_ERROR << "Failed to open file: " << filePath;
    return false;
  }

  std::ostringstream oss;
  oss << file.rdbuf();
  *content = oss.str();
  file.close();

  return true;
}

bool WriteFileContent(const std::string &filePath, const std::string &content) {
  std::ofstream file(filePath);
  if (!file.is_open()) {
    LOG_ERROR << "Failed to write file: " << filePath;
    return false;
  }

  file << content;
  file.close();

  return true;
}

bool ExecuteCommand(const std::string &command, std::string *output, int *exitCode) {
  // Open pipe to command
  FILE *pipe = popen((command + " 2>&1").c_str(), "r");
  if (pipe == nullptr) {
    LOG_ERROR << "Failed to execute command: " << command;
    *exitCode = -1;
    return false;
  }

  // Read command output
  char buffer[256];
  std::ostringstream oss;
  while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
    oss << buffer;
  }

  *output = oss.str();
  *exitCode = pclose(pipe);

  return true;
}

bool RunBishengirOpt(const std::string &inputFile, const std::string &outputFile,
                     const std::string &bishengirOptPath) {
  // Run bishengir-opt to convert linalg.generic to named operations
  std::ostringstream cmd;
  cmd << bishengirOptPath << " " << inputFile << " --hfusion-convert-generic-to-named" << " -o " << outputFile;

  LOG_OUT << "Bishengir opt command: " << cmd.str();

  std::string output;
  int exitCode;
  if (!ExecuteCommand(cmd.str(), &output, &exitCode)) {
    LOG_ERROR << "Failed to execute bishengir-opt";
    return false;
  }

  if (exitCode != 0) {
    LOG_ERROR << "bishengir-opt failed with exit code " << exitCode << ":\n" << output;
    return false;
  }

  LOG_OUT << "bishengir-opt completed: " << outputFile;
  return true;
}

bool RunBishengirCompile(const std::string &linalgFile, const std::string &outputSo,
                         const std::string &bishengirCompilePath) {
  // Get SoC name from ACL interface
  const char *socName = mrt::device::ascend::GetAscendSocVersion();
  if (socName == nullptr) {
    LOG_ERROR << "Failed to get SoC name from ACL";
    return false;
  }

  // Run bishengir-compile to generate .so from Linalg IR
  std::ostringstream cmd;
  cmd << bishengirCompilePath << " " << linalgFile << " --enable-hfusion-compile=true"
      << " --enable-hivm-compile=true" << " --enable-auto-multi-buffer=true" << " --target=" << socName
      << " --enable-bin-relocation=false" << " -o " << outputSo;

  LOG_OUT << "Bishengir compile command: " << cmd.str();

  std::string output;
  int exitCode;
  if (!ExecuteCommand(cmd.str(), &output, &exitCode)) {
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

bool ExtractFunctionNames(const std::string &mlirText, std::string *entryName) {
  // Extract function name from MLIR text using manual parsing
  // Look for pattern: func.func @function_name
  const std::string pattern = "func.func @";
  size_t pos = mlirText.find(pattern);

  if (pos == std::string::npos) {
    entryName->clear();
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
  while (pos < mlirText.size() && (std::isalnum(static_cast<unsigned char>(mlirText[pos])) || mlirText[pos] == '_')) {
    ++pos;
  }

  if (pos > start) {
    *entryName = mlirText.substr(start, pos - start);
    LOG_OUT << "Extracted function name: " << *entryName;
    return true;
  }

  entryName->clear();
  LOG_ERROR << "Failed to extract function name from MLIR";
  return false;
}

}  // namespace internal

CompileResult CompileFromText(const CompileRequest &request) {
  CompileResult result;

  if (request.mlirText.empty()) {
    result.errorMessage = "Empty MLIR text";
    LOG_ERROR << result.errorMessage;
    return result;
  }

  // Extract function names from MLIR text first (needed for cache directory name)
  std::string entryName;
  if (!internal::ExtractFunctionNames(request.mlirText, &entryName)) {
    result.errorMessage = "Failed to extract function name from MLIR";
    LOG_ERROR << result.errorMessage;
    return result;
  }

  // Generate unique ID for this compilation (content-based hash)
  std::string uniqueId = internal::ComputeHash(request.mlirText);

  // Create cache directory: cacheDir/entryName_hash (easier to identify during debugging)
  std::string compilationCacheDir = request.cacheDir + "/" + entryName + "_" + uniqueId;
  std::string mkdirCmd = "mkdir -p " + compilationCacheDir;
  int ret = system(mkdirCmd.c_str());
  if (ret != 0) {
    result.errorMessage = "Failed to create cache directory: " + compilationCacheDir;
    LOG_ERROR << result.errorMessage;
    return result;
  }

  LOG_OUT << "Compiling MLIR text (function: " << entryName << ", id: " << uniqueId << ")";
  LOG_OUT << "Cache directory: " << compilationCacheDir;

  // Create temporary files in the unique cache directory
  std::string inputFile = compilationCacheDir + "/" + entryName + "_" + uniqueId + "_input.mlir";
  std::string outputSo = compilationCacheDir + "/" + entryName + "_" + uniqueId + ".so";

  // Write MLIR text to file
  if (!internal::WriteFileContent(inputFile, request.mlirText)) {
    result.errorMessage = "Failed to write MLIR text to file";
    LOG_ERROR << result.errorMessage;
    return result;
  }

  // Run bishengir-opt to convert linalg.generic to named operations
  std::string optOutputFile = compilationCacheDir + "/" + entryName + "_" + uniqueId + "_opt.mlir";
  if (!internal::RunBishengirOpt(inputFile, optOutputFile, request.bishengirOptPath)) {
    result.errorMessage = "bishengir-opt failed";
    LOG_ERROR << result.errorMessage;
    return result;
  }

  // Run bishengir-compile on the optimized MLIR
  if (!internal::RunBishengirCompile(optOutputFile, outputSo, request.bishengirCompilePath)) {
    result.errorMessage = "bishengir-compile failed";
    LOG_ERROR << result.errorMessage;
    return result;
  }

  // bishengir-compile adds "lib" prefix to the output filename
  // If outputSo is "path/uniqueId.so", actual file is "path/libuniqueId.so"
  std::string actualSo;
  size_t lastSlash = outputSo.find_last_of('/');
  if (lastSlash != std::string::npos) {
    actualSo = outputSo.substr(0, lastSlash + 1) + "lib" + outputSo.substr(lastSlash + 1);
  } else {
    actualSo = "lib" + outputSo;
  }

  // Set function names
  result.entryName = entryName;
  LOG_OUT << "Entry function: " << result.entryName;

  // Clean up intermediate files if not keeping them
  if (!request.keepIntermediateFiles) {
    unlink(inputFile.c_str());
    unlink(optOutputFile.c_str());
  } else {
    LOG_OUT << "Intermediate files kept for debugging:";
    LOG_OUT << "  Input: " << inputFile;
    LOG_OUT << "  Optimized: " << optOutputFile;
  }

  // Fill result
  result.success = true;
  result.cacheDir = compilationCacheDir;  // Return the cache directory path
  result.soPath = actualSo;               // Use the actual .so path with "lib" prefix

  LOG_OUT << "MLIR compilation successful: " << actualSo;
  return result;
}

}  // namespace MlirCompiler

}  // namespace mrt::ops
