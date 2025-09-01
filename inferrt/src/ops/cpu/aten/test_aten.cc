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

#include <vector>

#include "ops/cpu/aten/test_aten.h"
// This file need to be deleted in the future.
namespace mrt {
namespace ops {
void TestAtenKernel::Init() {
  CHECK_IF_NULL(node_);
  input_.clear();
  for (auto &input : node_->inputs) {
    CHECK_IF_NULL(input->output);
    input_.emplace_back(input->output.get());
  }
  CHECK_IF_NULL(node_->output);
  output_ = node_->output.get();
}

void TestAtenKernel::InferShape() {
  CHECK_IF_NULL(operator_);
  node_->output = ir::MakeIntrusive<ir::Value>(
    ir::Tensor({-1}, ir::DataType::Type::Float32, hardware::Device(hardware::DeviceType::CPU, 0)));
  Init();
  LOG_OUT << "Begin InferShape for operator [" << ToStr(node_->op) << "], input=" << input_ << ", output=" << output_;
  if (operator_->InferShape(input_, output_) != SUCCESS) {
    LOG_EXCEPTION << "Infer shape failed for operator " << ToStr(node_->op);
  }
}

void TestAtenKernel::Resize() {
  // null
}

void TestAtenKernel::Launch() {
  CHECK_IF_NULL(operator_);
  Init();
  LOG_OUT << "Begin Launch for operator [" << ToStr(node_->op) << "], input=" << input_ << ", output=" << output_;
  if (operator_->Launch(input_, {}, output_, nullptr) != SUCCESS) {
    LOG_EXCEPTION << "Launch operator " << ToStr(node_->op) << " failed";
  }
}

DART_REGISTER_KERNEL_LIB("TestAten", TestAtenKernelLib);

}  // namespace ops
}  // namespace mrt
