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

#ifndef __KERNEL_KERNEL_LIB_H__
#define __KERNEL_KERNEL_LIB_H__

#include <string>
#include <functional>

#include "kernel/kernel.h"
#include "tensor/tensor.h"

namespace da {
namespace kernel {

// The register entry of new kernel lib.
#define DART_REGISTER_KERNEL_LIB(KERNEL_LIB_NAME, KERNEL_LIB_CLASS)                   \
  static const da::kernel::KernelLibRegistrar g_kernel_lib_##KERNEL_LIB_CLASS##_reg( \
    KERNEL_LIB_NAME, []() { return new (std::nothrow) KERNEL_LIB_CLASS(); });

class KernelLib;
using KernelLibCreator = std::function<KernelLib *()>;

class DA_API KernelLib {
 public:
  explicit KernelLib(const std::string &&name) : name_(std::move(name)) {}
  virtual ~KernelLib() = default;

  virtual DAKernel *CreateKernel(tensor::DATensor *tensorNode) const = 0;
  std::string Name() const { return name_; }

 protected:
  std::string name_;
};

class DA_API KernelLibRegistry {
 public:
  static KernelLibRegistry &Instance();

  void Register(const std::string &name, const KernelLibCreator &&creator);
  void Load(const std::string &path);
  const KernelLib *Get(const std::string &name);

 private:
  KernelLibRegistry() = default;
  ~KernelLibRegistry();

 private:
  std::unordered_map<std::string, const KernelLib *> kernelLibs_;
  std::unordered_map<std::string, const KernelLibCreator> kernelLibCreators_;
  std::unordered_map<std::string, void *> kernelLibHandles_;
};

class DA_API KernelLibRegistrar {
 public:
  KernelLibRegistrar(const std::string &name, const KernelLibCreator &&creator) {
    KernelLibRegistry::Instance().Register(name, std::move(creator));
  }
  ~KernelLibRegistrar() = default;
};

}  // namespace kernel
}  // namespace da
#endif  // __KERNEL_KERNEL_LIB_H__
