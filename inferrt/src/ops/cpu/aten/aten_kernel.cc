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

namespace da {
namespace ops {

// Common Aten utils
namespace {
at::ScalarType ToAtenDType(tensor::Type type) {
  switch (type) {
    case tensor::Type_Bool:
      return at::kBool;
    case tensor::Type_F16:
      return at::kHalf;
    case tensor::Type_F32:
      return at::kFloat;
    case tensor::Type_F64:
      return at::kDouble;
    case tensor::Type_I16:
      return at::kShort;
    case tensor::Type_I32:
      return at::kInt;
    case tensor::Type_I64:
      return at::kLong;
    case tensor::Type_BF16:
      return at::kBFloat16;
    default:
      LOG_ERROR << "Unsupported da::tensor::Type for Aten conversion.";
      exit(EXIT_FAILURE);
  }
}

at::Tensor ToAtenTensor(const tensor::DATensor *tensor) {
  CHECK_IF_NULL(tensor);
  CHECK_IF_NULL(tensor->data);
  std::vector<int64_t> shape(tensor->shape, tensor->shape + tensor->dim);
  auto options = at::TensorOptions().dtype(ToAtenDType(tensor->type));
  return at::from_blob(tensor->data, shape, options);
}

static void AllocateTensorData(tensor::DATensor *tensor) {
  size_t size = tensor::ShapeSize(tensor->shape) * tensor::DataTypeSize(tensor->type);
  tensor->data = malloc(size);
}
}  // namespace

// AtenKernel
void AtenKernel::InferShape() {
  if (tensorNode_->inputSize == 1) {
    runtime::CloneDATensorTypeAndShape(tensorNode_, tensorNode_->input[0]);
  } else if (tensorNode_->inputSize == 2) {
    auto in0 = tensorNode_->input[0];
    auto in1 = tensorNode_->input[1];

    std::vector<int64_t> in0_shape(in0->shape, in0->shape + in0->dim);
    std::vector<int64_t> in1_shape(in1->shape, in1->shape + in1->dim);
    auto out_shape = at::infer_size(in0_shape, in1_shape);

    tensorNode_->dim = out_shape.size();
    for (size_t i = 0; i < out_shape.size(); ++i) {
      tensorNode_->shape[i] = out_shape[i];
    }
  }
  LOG_OUT << "tensor shape after infer: " << ToString(tensorNode_);
}

void AtenKernel::Resize() {}

// Aten Kernels
#define IMPLEMENT_ATEN_UNARY_OP_OUT(Name, Op)      \
  void Aten##Name::Launch() {                      \
    auto in = ToAtenTensor(tensorNode_->input[0]); \
    AllocateTensorData(tensorNode_);               \
    auto out = ToAtenTensor(tensorNode_);          \
    at::Op##_out(out, in);                         \
  }

#define IMPLEMENT_ATEN_BINARY_OP_OUT(Name, Op)      \
  void Aten##Name::Launch() {                       \
    auto in0 = ToAtenTensor(tensorNode_->input[0]); \
    auto in1 = ToAtenTensor(tensorNode_->input[1]); \
    AllocateTensorData(tensorNode_);                \
    auto out = ToAtenTensor(tensorNode_);           \
    at::Op##_out(out, in0, in1);                    \
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
DAKernel *AtenKernelLib::CreateKernel(tensor::DATensor *tensorNode) const {
  switch (tensorNode->op) {
    case ops::Op_add:
      return new AtenAdd(tensorNode);
    case ops::Op_sub:
      return new AtenSub(tensorNode);
    case ops::Op_mul:
      return new AtenMul(tensorNode);
    case ops::Op_div:
      return new AtenDiv(tensorNode);
    case ops::Op_matmul:
      return new AtenMatmul(tensorNode);
    case ops::Op_neg:
      return new AtenNeg(tensorNode);
    case ops::Op_square:
      return new AtenSquare(tensorNode);
    case ops::Op_rsqrt:
      return new AtenRsqrt(tensorNode);
    case ops::Op_relu:
      return new AtenRelu(tensorNode);
    case ops::Op_sigmoid:
      return new AtenSigmoid(tensorNode);
    case ops::Op_gelu:
      return new AtenGelu(tensorNode);
    case ops::Op_silu:
      return new AtenSilu(tensorNode);
    default:
      LOG_OUT << "Unsupported op for Aten kernel lib: " << ops::ToStr(tensorNode->op);
      return nullptr;
  }
}

DART_REGISTER_KERNEL_LIB("Aten", AtenKernelLib);

}  // namespace ops
}  // namespace da
