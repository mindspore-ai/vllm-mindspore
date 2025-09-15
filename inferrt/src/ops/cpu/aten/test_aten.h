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

#ifndef __OPS_CPU_ATEN_TEST_ATEN_MATMUL_H__
#define __OPS_CPU_ATEN_TEST_ATEN_MATMUL_H__

#include <memory>
#include <vector>

#include "ops/op_def/ops_name.h"
#include "ops/operator.h"
#include "ops/kernel_lib.h"
#include "ops/op_register.h"
// This file need to be deleted in the future.
namespace mrt {
namespace ops {
class TestAtenKernel : public DAKernel {
 public:
  explicit TestAtenKernel(ir::NodePtr node) : DAKernel(node) {
    operator_ = CreateOperator(ToStr(node->op), hardware::DeviceType::CPU);
  }
  void Init() override;
  void InferShape() override;
  void Resize() override;
  void Launch() override;

 private:
  std::unique_ptr<Operator> operator_;
  std::vector<const ir::Value *> input_;
  ir::Value *output_;
};

class DA_API TestAtenKernelLib : public KernelLib {
 public:
  TestAtenKernelLib() : KernelLib("TestAten") {}
  ~TestAtenKernelLib() = default;
  DAKernel *CreateKernel(ir::NodePtr node) const override { return new TestAtenKernel(node); }
};
}  // namespace ops
}  // namespace mrt
#endif  // __OPS_CPU_ATEN_TEST_ATEN_MATMUL_H__
