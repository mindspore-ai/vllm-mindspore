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

#include "ops/utils/data_convert.h"

namespace mrt {
namespace ops {
constexpr size_t kRealInputOffset = 1;

void OpTorchCall::updateTorchTensor(at::Tensor &atTensor, const ir::TensorPtr &mrtTensor) {
  CHECK_IF_NULL(mrtTensor);
  auto tensor_impl = atTensor.unsafeGetTensorImpl();
  at::DataPtr dataPtr = at::DataPtr(const_cast<void *>(mrtTensor->DataPtr()), atTensor.device());
  atTensor.storage().unsafeGetStorageImpl()->set_data_ptr(std::move(dataPtr));
  auto shape = mrtTensor->Shape();
  auto strides = mrtTensor->Strides();
  if (strides.empty()) {
    tensor_impl->set_sizes_contiguous(shape);
  } else {
    tensor_impl->set_sizes_and_strides(shape, strides);
  }
}

// Helper functions for ConvertTupleToStack
void OpTorchCall::ConvertTensorTupleToStack(const ir::TuplePtr tuple, torch::jit::Stack &stack) {
  std::vector<at::Tensor> vec;
  vec.reserve(tuple->Size());
  for (size_t i = 0; i < tuple->Size(); i++) {
    if (firstRun_) {
      auto atTensor = ToTorchTensor((*tuple)[i]->ToTensor());
      vec.push_back(atTensor);
      atTensors_.push_back(atTensor);
    } else {
      CHECK_IF_FAIL(tensorIdx_ < atTensors_.size());
      updateTorchTensor(atTensors_[tensorIdx_], (*tuple)[i]->ToTensor());
      vec.push_back(atTensors_[tensorIdx_]);
    }
    tensorIdx_++;
  }
  torch::jit::push(stack, torch::jit::IValue(vec));
}

void OpTorchCall::ConvertIntTupleToStack(const ir::TuplePtr tuple, torch::jit::Stack &stack) {
  std::vector<int64_t> vec;
  vec.reserve(tuple->Size());
  for (size_t i = 0; i < tuple->Size(); i++) {
    vec.push_back((*tuple)[i]->ToInt());
  }
  torch::jit::push(stack, torch::jit::IValue(vec));
}

void OpTorchCall::ConvertBoolTupleToStack(const ir::TuplePtr tuple, torch::jit::Stack &stack) {
  std::vector<bool> vec;
  vec.reserve(tuple->Size());
  for (size_t i = 0; i < tuple->Size(); i++) {
    vec.push_back((*tuple)[i]->ToBool());
  }
  torch::jit::push(stack, torch::jit::IValue(vec));
}

void OpTorchCall::ConvertDoubleTupleToStack(const ir::TuplePtr tuple, torch::jit::Stack &stack) {
  std::vector<double> vec;
  vec.reserve(tuple->Size());
  for (size_t i = 0; i < tuple->Size(); i++) {
    vec.push_back((*tuple)[i]->ToDouble());
  }
  torch::jit::push(stack, torch::jit::IValue(vec));
}

void OpTorchCall::ConvertStringTupleToStack(const ir::TuplePtr tuple, torch::jit::Stack &stack) {
  std::vector<std::string> vec;
  vec.reserve(tuple->Size());
  for (size_t i = 0; i < tuple->Size(); i++) {
    vec.push_back((*tuple)[i]->ToString());
  }
  torch::jit::push(stack, torch::jit::IValue(vec));
}

void OpTorchCall::ConvertNoneTupleToStack(const ir::TuplePtr tuple, torch::jit::Stack &stack) {
  torch::jit::push(stack, torch::jit::IValue());
}

void OpTorchCall::ConvertTupleToStack(const ir::TuplePtr tuple, torch::jit::Stack &stack) {
  CHECK_IF_NULL(tuple);
  size_t size = tuple->Size();
  if (size == 0) {
    // Handle empty tuple if needed
    return;
  }

  auto element = (*tuple)[0].get();
  ir::Value::Tag tag = element->GetTag();

  // Define a map from Tag to tuple conversion function
  static const std::unordered_map<ir::Value::Tag, ConvertTupleFunc> tuple_conversion_map = {
    {ir::Value::Tag::Tensor, &OpTorchCall::ConvertTensorTupleToStack},
    {ir::Value::Tag::Int, &OpTorchCall::ConvertIntTupleToStack},
    {ir::Value::Tag::Symbol, &OpTorchCall::ConvertIntTupleToStack},
    {ir::Value::Tag::Bool, &OpTorchCall::ConvertBoolTupleToStack},
    {ir::Value::Tag::Double, &OpTorchCall::ConvertDoubleTupleToStack},
    {ir::Value::Tag::String, &OpTorchCall::ConvertStringTupleToStack},
    {ir::Value::Tag::None, &OpTorchCall::ConvertNoneTupleToStack}};

  auto it = tuple_conversion_map.find(tag);
  if (it != tuple_conversion_map.end()) {
    it->second(this, tuple, stack);
  } else {
    LOG_EXCEPTION << "Unsupported tuple element type: " << *element;
  }
}

// Helper functions for ConvertInputsToStack
void OpTorchCall::ConvertTensorInputToStack(const ir::Value *value, torch::jit::Stack &stack) {
  if (firstRun_) {
    auto atTensor = ToTorchTensor(value->ToTensor());
    atTensors_.push_back(atTensor);
    torch::jit::push(stack, atTensor);
  } else {
    // update trensor
    CHECK_IF_FAIL(tensorIdx_ < atTensors_.size());
    auto &atTensor = atTensors_[tensorIdx_];
    updateTorchTensor(atTensor, value->ToTensor());
    torch::jit::push(stack, atTensor);
  }
  tensorIdx_++;
}

void OpTorchCall::ConvertDoubleInputToStack(const ir::Value *value, torch::jit::Stack &stack) {
  torch::jit::push(stack, value->ToDouble());
}

void OpTorchCall::ConvertIntInputToStack(const ir::Value *value, torch::jit::Stack &stack) {
  torch::jit::push(stack, value->ToInt());
}

void OpTorchCall::ConvertBoolInputToStack(const ir::Value *value, torch::jit::Stack &stack) {
  torch::jit::push(stack, value->ToBool());
}

void OpTorchCall::ConvertStringInputToStack(const ir::Value *value, torch::jit::Stack &stack) {
  torch::jit::push(stack, value->ToString());
}

void OpTorchCall::ConvertTupleInputToStack(const ir::Value *value, torch::jit::Stack &stack) {
  ConvertTupleToStack(value->ToTuple(), stack);
}

void OpTorchCall::ConvertNoneInputToStack(const ir::Value *value, torch::jit::Stack &stack) {
  torch::jit::push(stack, torch::jit::IValue());
}

void OpTorchCall::ConvertInputsToStack(const std::vector<const ir::Value *> &inputs, torch::jit::Stack &stack) {
  // Define a map from Tag to conversion function
  static const std::unordered_map<ir::Value::Tag, ConvertInputsFunc> conversion_map = {
    {ir::Value::Tag::Tensor, &OpTorchCall::ConvertTensorInputToStack},
    {ir::Value::Tag::Double, &OpTorchCall::ConvertDoubleInputToStack},
    {ir::Value::Tag::Int, &OpTorchCall::ConvertIntInputToStack},
    {ir::Value::Tag::Symbol, &OpTorchCall::ConvertIntInputToStack},
    {ir::Value::Tag::Bool, &OpTorchCall::ConvertBoolInputToStack},
    {ir::Value::Tag::String, &OpTorchCall::ConvertStringInputToStack},
    {ir::Value::Tag::Tuple, &OpTorchCall::ConvertTupleInputToStack},
    {ir::Value::Tag::None, &OpTorchCall::ConvertNoneInputToStack}};

  for (auto value : inputs) {
    ir::Value::Tag tag = value->GetTag();
    auto it = conversion_map.find(tag);
    if (it != conversion_map.end()) {
      it->second(this, value, stack);
    } else {
      LOG_EXCEPTION << "Unsupported value type: " << *value;
    }
  }
}

void OpTorchCall::ToMrtTensor(ir::Value *output, torch::jit::IValue &&ivalue) const {
  if (ivalue.isTensor() && output->IsTensor()) {
    auto &tensor = ivalue.toTensor();
    auto &outTensor = output->ToTensor();
    if (!IsTorchTensorStandardLayout(tensor)) {
      LOG_EXCEPTION << "For '" << qualifiedOpName_ << "', The output tensor is not in standard layout.";
    }
    auto data_ptr = tensor.storage().set_data_ptr(std::move(c10::DataPtr()));  // return the original data ptr.
    auto deleter = data_ptr.get_deleter();
    auto *data = data_ptr.release_context();
    outTensor->GetStorage()->SetDataPtrFromAten(data, deleter);
  } else if (ivalue.isList()) {
    auto &tuple = output->ToTuple();
    CHECK_IF_NULL(tuple);
    auto list = ivalue.toList();
    if (list.size() != tuple->Size()) {
      LOG_EXCEPTION << "List size not match tuple size";
    }
    for (size_t i = 0; i < list.size(); i++) {
      ToMrtTensor((*tuple)[i].get(), torch::jit::IValue{list.get(i)});
    }
  } else if (output->IsSymbol()) {
    // If output is symbol, we just ignore it.
    return;
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

bool OpTorchCall::NeedLaunch() { return false; }

OpsErrorCode OpTorchCall::CalcWorkspace(const std::vector<const ir::Value *> &input, const ir::Value *output,
                                        size_t *workspaceSize) {
  torch::jit::Stack stack;
  tensorIdx_ = 0;
  // Inputs process, convert to aten tensor and push to stack.
  ConvertInputsToStack(input, stack);
  operation_(stack);
  // Outputs process. Convert aten tensor to ir::Value.
  ConvertStackToOutput(const_cast<ir::Value *>(output), std::move(stack));
  CheckOutputInputRef(input, output, qualifiedOpName_);
  firstRun_ = false;
  return SUCCESS;
}
}  // namespace ops
}  // namespace mrt
