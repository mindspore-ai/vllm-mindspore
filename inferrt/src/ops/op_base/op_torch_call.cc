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

#include "ops/op_base/op_torch_call.h"
#include <sstream>
#include <unordered_map>
#include <functional>

#ifdef ENABLE_TORCH_NPU
#include <torch_npu/csrc/core/npu/NPUStream.h>
#endif

#include "ops/utils/aten_convert.h"
#include "ops/utils/data_convert.h"

#define CONVERT_TUPLE_TO_STACK(func) [this](const ir::TuplePtr tuple, torch::jit::Stack &stack) { func(tuple, stack); }

#define CONVERT_INPUT_TO_STACK(func) [this](const ir::Value *value, torch::jit::Stack &stack) { func(value, stack); }

namespace mrt {
namespace ops {
constexpr size_t kRealInputOffset = 1;

// Helper functions for ConvertTupleToStack
void OpTorchCall::ConvertTensorTupleToStack(const ir::TuplePtr tuple, torch::jit::Stack &stack) const {
  std::vector<at::Tensor> vec;
  vec.reserve(tuple->Size());
  for (size_t i = 0; i < tuple->Size(); i++) {
    vec.push_back(ToAtenTensor((*tuple)[i].get()));
  }
  torch::jit::push(stack, torch::jit::IValue(vec));
}

void OpTorchCall::ConvertIntTupleToStack(const ir::TuplePtr tuple, torch::jit::Stack &stack) const {
  std::vector<int64_t> vec;
  vec.reserve(tuple->Size());
  for (size_t i = 0; i < tuple->Size(); i++) {
    vec.push_back((*tuple)[i]->ToInt());
  }
  torch::jit::push(stack, torch::jit::IValue(vec));
}

void OpTorchCall::ConvertBoolTupleToStack(const ir::TuplePtr tuple, torch::jit::Stack &stack) const {
  std::vector<bool> vec;
  vec.reserve(tuple->Size());
  for (size_t i = 0; i < tuple->Size(); i++) {
    vec.push_back((*tuple)[i]->ToBool());
  }
  torch::jit::push(stack, torch::jit::IValue(vec));
}

void OpTorchCall::ConvertDoubleTupleToStack(const ir::TuplePtr tuple, torch::jit::Stack &stack) const {
  std::vector<double> vec;
  vec.reserve(tuple->Size());
  for (size_t i = 0; i < tuple->Size(); i++) {
    vec.push_back((*tuple)[i]->ToDouble());
  }
  torch::jit::push(stack, torch::jit::IValue(vec));
}

void OpTorchCall::ConvertStringTupleToStack(const ir::TuplePtr tuple, torch::jit::Stack &stack) const {
  std::vector<std::string> vec;
  vec.reserve(tuple->Size());
  for (size_t i = 0; i < tuple->Size(); i++) {
    vec.push_back((*tuple)[i]->ToString());
  }
  torch::jit::push(stack, torch::jit::IValue(vec));
}

void OpTorchCall::ConvertNoneTupleToStack(const ir::TuplePtr tuple, torch::jit::Stack &stack) const {
  torch::jit::push(stack, torch::jit::IValue());
}

void OpTorchCall::ConvertTupleToStack(const ir::TuplePtr tuple, torch::jit::Stack &stack) const {
  CHECK_IF_NULL(tuple);
  size_t size = tuple->Size();
  if (size == 0) {
    // Handle empty tuple if needed
    return;
  }

  auto element = (*tuple)[0].get();
  ir::Value::Tag tag = element->GetTag();

  // Define a map from Tag to tuple conversion function
  static const std::unordered_map<ir::Value::Tag, std::function<void(const ir::TuplePtr, torch::jit::Stack &)>>
    tuple_conversion_map = {{ir::Value::Tag::Tensor, CONVERT_TUPLE_TO_STACK(ConvertTensorTupleToStack)},
                            {ir::Value::Tag::Int, CONVERT_TUPLE_TO_STACK(ConvertIntTupleToStack)},
                            {ir::Value::Tag::Symbol, CONVERT_TUPLE_TO_STACK(ConvertIntTupleToStack)},
                            {ir::Value::Tag::Bool, CONVERT_TUPLE_TO_STACK(ConvertBoolTupleToStack)},
                            {ir::Value::Tag::Double, CONVERT_TUPLE_TO_STACK(ConvertDoubleTupleToStack)},
                            {ir::Value::Tag::String, CONVERT_TUPLE_TO_STACK(ConvertStringTupleToStack)},
                            {ir::Value::Tag::None, CONVERT_TUPLE_TO_STACK(ConvertNoneTupleToStack)}};

  auto it = tuple_conversion_map.find(tag);
  if (it != tuple_conversion_map.end()) {
    it->second(tuple, stack);
  } else {
    LOG_EXCEPTION << "Unsupported tuple element type: " << *element;
  }
}

// Helper functions for ConvertInputsToStack
void OpTorchCall::ConvertTensorInputToStack(const ir::Value *value, torch::jit::Stack &stack) const {
  torch::jit::push(stack, ToAtenTensor(value));
}

void OpTorchCall::ConvertDoubleInputToStack(const ir::Value *value, torch::jit::Stack &stack) const {
  torch::jit::push(stack, value->ToDouble());
}

void OpTorchCall::ConvertIntInputToStack(const ir::Value *value, torch::jit::Stack &stack) const {
  torch::jit::push(stack, value->ToInt());
}

void OpTorchCall::ConvertBoolInputToStack(const ir::Value *value, torch::jit::Stack &stack) const {
  torch::jit::push(stack, value->ToBool());
}

void OpTorchCall::ConvertStringInputToStack(const ir::Value *value, torch::jit::Stack &stack) const {
  torch::jit::push(stack, value->ToString());
}

void OpTorchCall::ConvertTupleInputToStack(const ir::Value *value, torch::jit::Stack &stack) const {
  ConvertTupleToStack(value->ToTuple(), stack);
}

void OpTorchCall::ConvertNoneInputToStack(const ir::Value *value, torch::jit::Stack &stack) const {
  torch::jit::push(stack, torch::jit::IValue());
}

void OpTorchCall::ConvertInputsToStack(const std::vector<const ir::Value *> &inputs, torch::jit::Stack &stack) const {
  // Define a map from Tag to conversion function
  static const std::unordered_map<ir::Value::Tag, std::function<void(const ir::Value *, torch::jit::Stack &)>>
    conversion_map = {{ir::Value::Tag::Tensor, CONVERT_INPUT_TO_STACK(ConvertTensorInputToStack)},
                      {ir::Value::Tag::Double, CONVERT_INPUT_TO_STACK(ConvertDoubleInputToStack)},
                      {ir::Value::Tag::Int, CONVERT_INPUT_TO_STACK(ConvertIntInputToStack)},
                      {ir::Value::Tag::Symbol, CONVERT_INPUT_TO_STACK(ConvertIntInputToStack)},
                      {ir::Value::Tag::Bool, CONVERT_INPUT_TO_STACK(ConvertBoolInputToStack)},
                      {ir::Value::Tag::String, CONVERT_INPUT_TO_STACK(ConvertStringInputToStack)},
                      {ir::Value::Tag::Tuple, CONVERT_INPUT_TO_STACK(ConvertTupleInputToStack)},
                      {ir::Value::Tag::None, CONVERT_INPUT_TO_STACK(ConvertNoneInputToStack)}};

  for (auto value : inputs) {
    ir::Value::Tag tag = value->GetTag();
    auto it = conversion_map.find(tag);
    if (it != conversion_map.end()) {
      it->second(value, stack);
    } else {
      LOG_EXCEPTION << "Unsupported value type: " << *value;
    }
  }
}

void OpTorchCall::ToMrtTensor(ir::Value *output, torch::jit::IValue ivalue) const {
  if (ivalue.isTensor() && output->IsTensor()) {
    auto tensor = ivalue.toTensor();
    auto outTensor = output->ToTensor();
    std::vector<int64_t> atenShape(tensor.sizes().begin(), tensor.sizes().end());
    if (atenShape != outTensor->Shape()) {
      LOG_EXCEPTION << "For '" << qualifiedOpName_
                    << "', The output tensor shape not match, expect: " << outTensor->Shape()
                    << ", but got: " << atenShape;
    }
    auto data_ptr = tensor.storage().set_data_ptr(std::move(c10::DataPtr()));  // return the original data ptr.
    void *data = data_ptr.get();
    auto data_ptr_shared = std::make_shared<c10::DataPtr>(std::move(data_ptr));
    auto deleter = [data_ptr_shared](void *) mutable {
      if (data_ptr_shared) {
        data_ptr_shared->clear();
      }
    };
    outTensor->GetStorage()->SetDataPtrFromAten(data, deleter);
  } else if (ivalue.isList()) {
    auto &tuple = output->ToTuple();
    CHECK_IF_NULL(tuple);
    auto list = std::move(ivalue).toList();
    if (list.size() != tuple->Size()) {
      LOG_EXCEPTION << "List size not match tuple size";
    }
    for (size_t i = 0; i < list.size(); i++) {
      ToMrtTensor((*tuple)[i].get(), torch::jit::IValue{list.get(i)});
    }
  } else {
    LOG_EXCEPTION << "Output Only Support Tensor or List[Tensor], but got type: "
                  << c10::typeKindToString(ivalue.type()->kind());
  }
}

void OpTorchCall::ConvertStackToOutput(ir::Value *output, torch::jit::Stack &&stack) const {
  if (stack.empty()) {
    return;
  }

  if (stack.size() == 1) {
    ToMrtTensor(output, std::move(stack[0]));
    return;
  }

  auto &tuple = output->ToTuple();
  if (tuple->Size() != stack.size()) {
    LOG_EXCEPTION << "Tuple size not match stack size";
  }
  for (size_t i = 0; i < stack.size(); i++) {
    ToMrtTensor((*tuple)[i].get(), std::move(stack[i]));
  }
}

bool OpTorchCall::MatchOpSchema(const std::vector<const ir::Value *> &inputs,
                                const std::shared_ptr<torch::jit::Operator> op) const {
  auto args = op->schema().arguments();
  // First input is op name
  if (args.size() != inputs.size() - kRealInputOffset) {
    return false;
  }
  // Define type kind to check function map
  static const std::unordered_map<c10::TypeKind, std::function<bool(const ir::Value *)>> typeCheckMap = {
    {c10::TypeKind::TensorType, [](const ir::Value *val) { return val->IsTensor(); }},
    {c10::TypeKind::NumberType, [](const ir::Value *val) { return val->IsDouble() || val->IsInt(); }},
    {c10::TypeKind::IntType, [](const ir::Value *val) { return val->IsInt() || val->IsSymbol(); }},
    {c10::TypeKind::BoolType, [](const ir::Value *val) { return val->IsBool(); }},
    {c10::TypeKind::FloatType, [](const ir::Value *val) { return val->IsDouble(); }},
    {c10::TypeKind::StringType, [](const ir::Value *val) { return val->IsString(); }},
    {c10::TypeKind::TupleType, [](const ir::Value *val) { return val->IsTuple(); }},
    {c10::TypeKind::ListType, [](const ir::Value *val) { return val->IsTuple(); }},
    {c10::TypeKind::NoneType, [](const ir::Value *val) { return val->IsNone(); }}};

  for (size_t i = 0, j = kRealInputOffset; i < args.size(); ++i, ++j) {
    auto type = args[i].type();
    if (type->kind() == c10::TypeKind::OptionalType) {
      if (inputs[j]->IsNone()) {
        continue;
      }
      type = type->castRaw<c10::OptionalType>()->getElementType();
    }

    // Check if type kind is in the map
    auto it = typeCheckMap.find(type->kind());
    bool match = (it != typeCheckMap.end()) ? it->second(inputs[j]) : false;
    if (!match) {
      return false;
    }
  }
  return true;
}

std::string OpTorchCall::GetOpsExpr(const std::vector<const ir::Value *> &inputs) const {
  std::string expr = qualifiedOpName_ + "(";
  for (size_t i = kRealInputOffset; i < inputs.size(); ++i) {
    ir::Value::Tag tag = inputs[i]->GetTag();
    expr += TagToString(tag);
    if (i < inputs.size() - 1) {
      expr += ", ";
    }
  }
  expr += ")";
  return expr;
}

std::string OpTorchCall::GetAvailableTorchOps() const {
  auto ops = torch::jit::getAllOperatorsFor(torch::jit::Symbol::fromQualString(qualifiedOpName_));
  std::stringstream opsStr;
  for (auto op : ops) {
    opsStr << op->schema() << "\n";
  }
  return opsStr.str();
}

void OpTorchCall::Init(const std::vector<const ir::Value *> &inputs, const ir::Value *output) {
  auto ops = torch::jit::getAllOperatorsFor(torch::jit::Symbol::fromQualString(qualifiedOpName_));
  for (auto &op : ops) {
    if (MatchOpSchema(inputs, op)) {
      operation_ = op->getOperation();
      break;
    }
  }
  if (!operation_) {
    LOG_EXCEPTION << "Operator: " << GetOpsExpr(inputs) << " not found in torch plugin. The available operators are:\n"
                  << GetAvailableTorchOps();
  }
}

OpsErrorCode OpTorchCall::Launch(const std::vector<const ir::Value *> &input, void *workspace, size_t workspaceSize,
                                 ir::Value *output, void *stream) {
  return SUCCESS;
}

OpsErrorCode OpTorchCall::CalcWorkspace(const std::vector<const ir::Value *> &input, const ir::Value *output,
                                        size_t *workspaceSize) {
  torch::jit::Stack stack;
  // Inputs process, convert to aten tensor and push to stack.
  ConvertInputsToStack(input, stack);
  operation_(stack);
  // Outputs process. Convert aten tensor to ir::Value.
  ConvertStackToOutput(const_cast<ir::Value *>(output), std::move(stack));
  CheckOutputInputRef(input, output, qualifiedOpName_);
  return SUCCESS;
}
}  // namespace ops
}  // namespace mrt
