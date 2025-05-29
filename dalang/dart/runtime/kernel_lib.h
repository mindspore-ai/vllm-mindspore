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

#ifndef __RUNTIME_KERNEL_LIB_H__
#define __RUNTIME_KERNEL_LIB_H__

#include <string>
#include <memory>
#include <functional>
#include <unordered_map>

#include "runtime/mempool.h"
#include "common/common.h"
#include "tensor/tensor.h"

namespace da {
namespace runtime {
// The register entry of new kernel lib.
#define DART_REGISTER_KERNEL_LIB(KERNEL_LIB_NAME, KERNEL_LIB_CLASS)                   \
  static const da::runtime::KernelLibRegistrar g_kernel_lib_##KERNEL_LIB_CLASS##_reg( \
    KERNEL_LIB_NAME, []() { return new (std::nothrow) KERNEL_LIB_CLASS(); });
class KernelLib;
using KernelLibCreator = std::function<KernelLib *()>;

class KernelLib {
 public:
  KernelLib(std::string name) : name_(name) {}
  virtual ~KernelLib() = default;

  virtual bool RunTensor(tensor::DATensor *tensorNode, runtime::MemoryPool *mempool) const = 0;
  std::string Name() const { return name_; }

 protected:
  std::string name_;
};

class DA_API KernelLibRegistry {
 public:
  static KernelLibRegistry &Instance() {
    static KernelLibRegistry instance;
    return instance;
  }

  virtual ~KernelLibRegistry() {
    for (auto &iter : kernel_libs_) {
      delete iter.second;
    }
    kernel_libs_.clear();
  }

  void Register(const std::string &name, KernelLibCreator &&creator) {
    if (kernel_lib_creators_.find(name) == kernel_lib_creators_.end()) {
      LOG_OUT << "KernelLibCreator for " << name << " registered.";
      (void)kernel_lib_creators_.emplace(name, std::move(creator));
    }
  }

  const KernelLib *Get(const std::string &name) {
    if (auto iter = kernel_libs_.find(name); iter != kernel_libs_.end()) {
      return iter->second;
    }
    if (auto iter = kernel_lib_creators_.find(name); iter != kernel_lib_creators_.end()) {
      auto kernel_lib = (iter->second)();
      kernel_libs_[name] = kernel_lib;
      return kernel_lib;
    }
    LOG_ERROR << "KernelLib " << name << " is not exist.";
    return nullptr;
  }

 private:
  KernelLibRegistry() = default;

 private:
  std::unordered_map<std::string, const KernelLib *> kernel_libs_;
  std::unordered_map<std::string, KernelLibCreator> kernel_lib_creators_;
};

class KernelLibRegistrar {
 public:
  KernelLibRegistrar(const std::string &name, KernelLibCreator &&creator) {
    KernelLibRegistry::Instance().Register(name, std::move(creator));
  }
};

}  // namespace runtime
}  // namespace da
#endif  // __RUNTIME_KERNEL_LIB_H__
