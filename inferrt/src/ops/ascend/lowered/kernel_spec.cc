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

#include "ops/ascend/lowered/kernel_spec.h"
#include <utility>
#include "ops/ascend/lowered/mlir_compiler.h"

namespace mrt::ops {

// Internal default compiler callback
static bool InternalDefaultMlirCompiler(const std::string &mlirInput, std::string &outputSoPath,
                                        std::string &entryName, std::string &tilingPrefix) {
  MlirCompiler::CompileResult result = MlirCompiler::Instance().CompileFromText(mlirInput);
  if (!result.success) {
    return false;
  }
  outputSoPath = result.soPath;
  entryName = result.entryName;
  tilingPrefix = result.tilingPrefix;
  return true;
}

// KernelSpec constructor with default compiler
KernelSpec::KernelSpec(const std::string &id_, const std::string &mlirText_)
    : id(id_), mlirText(mlirText_), compiler(InternalDefaultMlirCompiler) {}

KernelRegistry &KernelRegistry::Instance() {
  static KernelRegistry instance;
  return instance;
}

bool KernelRegistry::Register(const std::string &specId, const std::string &mlirText) {
  std::lock_guard<std::mutex> lock(mutex_);

  // try_emplace returns pair<iterator, bool>, second element is true if insertion happened
  auto [it, inserted] = specs_.try_emplace(specId, specId, mlirText);
  return inserted;
}

const KernelSpec *KernelRegistry::Lookup(const std::string &specId) const {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = specs_.find(specId);
  if (it != specs_.end()) {
    return &(it->second);
  }

  return nullptr;
}

bool KernelRegistry::Contains(const std::string &specId) const {
  std::lock_guard<std::mutex> lock(mutex_);
  return specs_.find(specId) != specs_.end();
}

}  // namespace mrt::ops
