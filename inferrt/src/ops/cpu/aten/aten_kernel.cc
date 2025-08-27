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

#include <torch/torch.h>

#include "ops/op_def/ops_name.h"
#include "common/logger.h"
#include "runtime/utils/utils.h"

#include "ops/cpu/aten/aten_kernel.h"

namespace mrt {
namespace ops {

// Common Aten utils
namespace {
at::ScalarType ToAtenDType(ir::DataType type) {
  switch (type) {
    case ir::DataType::Bool:
      return at::kBool;
    case ir::DataType::Float32:
      return at::kFloat;
    case ir::DataType::Float64:
      return at::kDouble;
    case ir::DataType::Int16:
      return at::kShort;
    case ir::DataType::Int32:
      return at::kInt;
    case ir::DataType::Int64:
      return at::kLong;
    default:
      LOG_ERROR << "Unsupported DataType for Aten conversion.";
      exit(EXIT_FAILURE);
  }
}

at::Tensor ToAtenTensor(ir::ValuePtr value) {
  auto tensor = value->ToTensor();
  auto options = at::TensorOptions().dtype(ToAtenDType(tensor->Dtype()));
  return at::from_blob(tensor->DataPtr(), tensor->Shape(), options);
}
}  // namespace

// AtenKernel
void AtenKernel::InferShape() {
  std::vector<int64_t> dims;
  auto dtype = node_->inputs[0]->output->ToTensor()->Dtype();
  if (node_->inputs.size() == 1) {
    dims = node_->inputs[0]->output->ToTensor()->Shape();
  } else if (node_->inputs.size() == 2) {
    auto in0Dims = node_->inputs[0]->output->ToTensor()->Shape();
    auto in1Dims = node_->inputs[1]->output->ToTensor()->Shape();
    dims = at::infer_size(in0Dims, in1Dims);
  }
  node_->output = ir::MakeIntrusive<ir::Value>(ir::Tensor(dims, dtype, hardware::Device(hardware::DeviceType::CPU, 0)));
}

void AtenKernel::Resize() {}

// Aten Kernels
#define IMPLEMENT_ATEN_UNARY_OP_OUT(Name, Op)         \
  void Aten##Name::Launch() {                         \
    auto in = ToAtenTensor(node_->inputs[0]->output); \
    auto out = ToAtenTensor(node_->output);           \
    at::Op##_out(out, in);                            \
  }

#define IMPLEMENT_ATEN_BINARY_OP_OUT(Name, Op)         \
  void Aten##Name::Launch() {                          \
    auto in0 = ToAtenTensor(node_->inputs[0]->output); \
    auto in1 = ToAtenTensor(node_->inputs[1]->output); \
    auto out = ToAtenTensor(node_->output);            \
    at::Op##_out(out, in0, in1);                       \
  }

IMPLEMENT_ATEN_BINARY_OP_OUT(Add, add)
IMPLEMENT_ATEN_BINARY_OP_OUT(Sub, sub)
IMPLEMENT_ATEN_BINARY_OP_OUT(Mul, mul)
IMPLEMENT_ATEN_BINARY_OP_OUT(Div, div)
IMPLEMENT_ATEN_BINARY_OP_OUT(Matmul, matmul)

IMPLEMENT_ATEN_UNARY_OP_OUT(Neg, neg)
IMPLEMENT_ATEN_UNARY_OP_OUT(Square, square)
IMPLEMENT_ATEN_UNARY_OP_OUT(Rsqrt, rsqrt)
IMPLEMENT_ATEN_UNARY_OP_OUT(Relu, relu)
IMPLEMENT_ATEN_UNARY_OP_OUT(Sigmoid, sigmoid)
IMPLEMENT_ATEN_UNARY_OP_OUT(Gelu, gelu)
IMPLEMENT_ATEN_UNARY_OP_OUT(Silu, silu)

// AtenKernelLib
DAKernel *AtenKernelLib::CreateKernel(ir::NodePtr node) const {
  switch (node->op) {
    case ops::Op_add:
      return new AtenAdd(node);
    case ops::Op_sub:
      return new AtenSub(node);
    case ops::Op_mul:
      return new AtenMul(node);
    case ops::Op_div:
      return new AtenDiv(node);
    case ops::Op_matmul:
      return new AtenMatmul(node);
    case ops::Op_neg:
      return new AtenNeg(node);
    case ops::Op_square:
      return new AtenSquare(node);
    case ops::Op_rsqrt:
      return new AtenRsqrt(node);
    case ops::Op_relu:
      return new AtenRelu(node);
    case ops::Op_sigmoid:
      return new AtenSigmoid(node);
    case ops::Op_gelu:
      return new AtenGelu(node);
    case ops::Op_silu:
      return new AtenSilu(node);
    default:
      LOG_OUT << "Unsupported op for Aten kernel lib: " << node;
      return nullptr;
  }
}

DART_REGISTER_KERNEL_LIB("Aten", AtenKernelLib);

}  // namespace ops
}  // namespace mrt
