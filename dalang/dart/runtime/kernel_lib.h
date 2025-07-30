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
#ifndef _WIN32
#include <libgen.h>
#include <dlfcn.h>
#else
#include <windows.h>
#endif

#include "dart/runtime/kernel.h"
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
  KernelLib(std::string &&name) : name_(std::move(name)) {}
  virtual ~KernelLib() = default;

  virtual DAKernel* CreateKernel(tensor::DATensor *tensorNode) const = 0;
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

  ~KernelLibRegistry() {
    for (auto &iter : kernelLibs_) {
      delete iter.second;
    }
    kernelLibs_.clear();
  }

  void Register(const std::string &name, KernelLibCreator &&creator) {
    if (kernelLibCreators_.find(name) == kernelLibCreators_.end()) {
      LOG_OUT << "KernelLibCreator for " << name << " registered.";
      (void)kernelLibCreators_.emplace(name, std::move(creator));
    }
  }

  void Load(const std::string &path) {
    if (kernelLibHandles_.count(path) > 0) {
      return;
    }

    LOG_OUT << "Load kernel lib path: " << path;
    void *handle;
    std::string errMsg = "";
#ifndef _WIN32
    handle = dlopen(path.c_str(), RTLD_LAZY);
    const char *result = dlerror();
    errMsg = (result == nullptr) ? "Unknown" : result;
#else
    handle = LoadLibrary(path.c_str());
    errMsg = std::to_string(GetLastError());
#endif

    if (handle == nullptr) {
      LOG_ERROR << "Load " + path + " failed, error: " + errMsg;
      return;
    }
    (void)kernelLibHandles_.emplace(path, handle);
  }

  const KernelLib *Get(const std::string &name) {
    if (auto iter = kernelLibs_.find(name); iter != kernelLibs_.end()) {
      return iter->second;
    }
    if (auto iter = kernelLibCreators_.find(name); iter != kernelLibCreators_.end()) {
      auto kernelLib = (iter->second)();
      kernelLibs_[name] = kernelLib;
      return kernelLib;
    }
    LOG_ERROR << "KernelLib " << name << " is not exist.";
    return nullptr;
  }

 private:
  KernelLibRegistry() = default;

 private:
  std::unordered_map<std::string, const KernelLib *> kernelLibs_;
  std::unordered_map<std::string, KernelLibCreator> kernelLibCreators_;
  std::unordered_map<std::string, void *> kernelLibHandles_;
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
