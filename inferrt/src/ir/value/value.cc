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

#include <iostream>
#include <vector>
#include "ir/value/value.h"

namespace mrt {
namespace ir {

Value::Value(Tensor &&v) : tag_(Tag::Tensor) { new (&tensor_) Tensor(std::move(v)); }
Value::Value(double v) : tag_(Tag::Double), double_(v) {}
Value::Value(int64_t v) : tag_(Tag::Int), int_(v) {}
Value::Value(bool v) : tag_(Tag::Bool), bool_(v) {}
Value::Value(std::string &&v) : tag_(Tag::String) { new (&string_) std::string(std::move(v)); }
Value::Value(Tuple &&v) : tag_(Tag::Tuple) { new (&tuple_) Tuple(std::move(v)); }

Value::~Value() {
  switch (tag_) {
    case Tag::Tensor:
      tensor_.~Tensor();
      break;
    case Tag::String:
      string_.~basic_string();
      break;
    case Tag::Tuple:
      tuple_.~Tuple();
      break;
    default:
      break;
  }
}

#define CHECK_TAG(expected)              \
  if (tag_ != expected) {                \
    LOG_EXCEPTION << "Bad Value access"; \
  }

const Tensor *Value::ToTensor() const {
  CHECK_TAG(Tag::Tensor);
  return &tensor_;
}
Tensor *Value::ToTensor() {
  CHECK_TAG(Tag::Tensor);
  return &tensor_;
}
double Value::ToDouble() const {
  CHECK_TAG(Tag::Double);
  return double_;
}
int64_t Value::ToInt() const {
  CHECK_TAG(Tag::Int);
  return int_;
}
bool Value::ToBool() const {
  CHECK_TAG(Tag::Bool);
  return bool_;
}
const std::string *Value::ToString() const {
  CHECK_TAG(Tag::String);
  return &string_;
}
std::string *Value::ToString() {
  CHECK_TAG(Tag::String);
  return &string_;
}
const Tuple *Value::ToTuple() const {
  CHECK_TAG(Tag::Tuple);
  return &tuple_;
}
Tuple *Value::ToTuple() {
  CHECK_TAG(Tag::Tuple);
  return &tuple_;
}

std::ostream &operator<<(std::ostream &os, const Tuple *tuple) {
  os << "Tuple(";
  const auto tupleSize = tuple->Size();
  for (size_t i = 0; i < tupleSize; ++i) {
    os << (*tuple)[i];
    if (i < tupleSize - 1) {
      os << ", ";
    }
  }
  os << ")";
  return os;
}

std::ostream &operator<<(std::ostream &os, const ValuePtr &value) { return operator<<(os, value.get()); }

std::ostream &operator<<(std::ostream &os, Value *value) {
  if (value == nullptr) {
    os << "Null";
  } else {
    os << *value;
  }
  return os;
}

std::ostream &operator<<(std::ostream &os, const Value *value) {
  if (value == nullptr) {
    os << "Null";
  } else {
    os << *value;
  }
  return os;
}

std::ostream &operator<<(std::ostream &os, const std::vector<const Value *> &values) {
  os << "std::vector{";
  for (size_t i = 0; i < values.size(); ++i) {
    os << values[i];
    if (i < values.size() - 1) {
      os << ", ";
    }
  }
  os << "}";
  return os;
}

std::ostream &operator<<(std::ostream &os, const Value &value) {
  switch (value.tag_) {
    case Value::Tag::None:
      os << "None";
      break;
    case Value::Tag::Tensor:
      os << value.ToTensor();
      break;
    case Value::Tag::Double:
      os << value.ToDouble();
      break;
    case Value::Tag::Int:
      os << value.ToInt();
      break;
    case Value::Tag::Bool:
      os << (value.ToBool() ? "true" : "false");
      break;
    case Value::Tag::String:
      os << "\"" << value.ToString() << "\"";
      break;
    case Value::Tag::Tuple:
      os << value.ToTuple();
      break;
  }
  return os;
}

}  // namespace ir
}  // namespace mrt
