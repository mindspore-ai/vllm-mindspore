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

#ifndef __KERNELS_ATEN_ATEN_KERNELS_H__
#define __KERNELS_ATEN_ATEN_KERNELS_H__

#include "kernels/kernel.h"
#include "kernels/kernel_lib.h"

namespace da {
namespace kernels {

// Aten Kernel lib
class DA_API AtenKernelLib : public KernelLib {
 public:
  AtenKernelLib() : KernelLib("Aten") {}
  ~AtenKernelLib() = default;
  DAKernel *CreateKernel(tensor::DATensor *tensorNode) const override;
};

// Base class for Aten kernels
class AtenKernel : public DAKernel {
 public:
  using DAKernel::DAKernel;
  void Init() override {}
  void InferShape() override;
  void Resize() override;
  void Launch() override = 0;
};

#define DEFINE_ATEN_KERNEL(Name)                \
  class DA_API Aten##Name : public AtenKernel { \
   public:                                      \
    using AtenKernel::AtenKernel;               \
    void Launch() override;                     \
  };

DEFINE_ATEN_KERNEL(Add)
DEFINE_ATEN_KERNEL(Sub)
DEFINE_ATEN_KERNEL(Mul)
DEFINE_ATEN_KERNEL(Div)
DEFINE_ATEN_KERNEL(Matmul)
DEFINE_ATEN_KERNEL(Neg)
DEFINE_ATEN_KERNEL(Square)
DEFINE_ATEN_KERNEL(Rsqrt)
DEFINE_ATEN_KERNEL(Relu)
DEFINE_ATEN_KERNEL(Sigmoid)
DEFINE_ATEN_KERNEL(Gelu)
DEFINE_ATEN_KERNEL(Silu)

}  // namespace kernels
}  // namespace da

#endif  // __KERNELS_ATEN_ATEN_KERNELS_H__
