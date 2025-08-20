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

#include "kernel/kernel.h"
#include "kernel/kernel_lib.h"

namespace da {
namespace kernel {

// Dummy kernel
class DummyKernel : public DAKernel {
 public:
  using DAKernel::DAKernel;
  void Init() override {}
  void InferShape() override {}
  void Resize() override {}
  void Launch() override {}
};

// Dummy Kernel lib
class DA_API DummyKernelLib : public KernelLib {
 public:
  DummyKernelLib() : KernelLib("Dummy") {}
  ~DummyKernelLib() = default;
  DAKernel *CreateKernel(tensor::DATensor *tensorNode) const override { return new DummyKernel(tensorNode); }
};

DART_REGISTER_KERNEL_LIB("Dummy", DummyKernelLib);

}  // namespace kernel
}  // namespace da
