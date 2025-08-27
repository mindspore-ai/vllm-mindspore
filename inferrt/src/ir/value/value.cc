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

#include "ir/value/value.h"
#include <iostream>

namespace mrt {
namespace ir {

Value::Value(Tensor v) : tag_(Tag::Tensor), tensor_(std::move(v)) {}
Value::Value(double v) : tag_(Tag::Double), double_(v) {}
Value::Value(int64_t v) : tag_(Tag::Int), int_(v) {}
Value::Value(bool v) : tag_(Tag::Bool), bool_(v) {}
Value::Value(std::string v) : tag_(Tag::String), string_(std::move(v)) {}
Value::Value(Tuple v) : tag_(Tag::Tuple), tuple_(std::move(v)) {}

Value::~Value() { Destroy(); }

/**
 * @brief Destroys the currently held value by explicitly calling its destructor.
 * This is necessary because the value is stored in a union.
 */
void Value::Destroy() {
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

/**
 * @brief Copies the value from another Value object using placement new.
 * @param other The Value to copy from.
 */
void Value::CopyFrom(const Value &other) {
  tag_ = other.tag_;
  switch (tag_) {
    case Tag::Tensor:
      new (&tensor_) Tensor(other.tensor_);
      break;
    case Tag::Double:
      double_ = other.double_;
      break;
    case Tag::Int:
      int_ = other.int_;
      break;
    case Tag::Bool:
      bool_ = other.bool_;
      break;
    case Tag::String:
      new (&string_) std::string(other.string_);
      break;
    case Tag::Tuple:
      new (&tuple_) Tuple(other.tuple_);
      break;
    case Tag::None:
      break;
  }
}

Value::Value(const Value &other) { CopyFrom(other); }

Value &Value::operator=(const Value &other) {
  if (this != &other) {
    Destroy();
    CopyFrom(other);
  }
  return *this;
}

Value::Value(Value &&other) noexcept : tag_(other.tag_) {
  switch (tag_) {
    case Tag::Tensor:
      new (&tensor_) Tensor(std::move(other.tensor_));
      break;
    case Tag::Double:
      double_ = other.double_;
      break;
    case Tag::Int:
      int_ = other.int_;
      break;
    case Tag::Bool:
      bool_ = other.bool_;
      break;
    case Tag::String:
      new (&string_) std::string(std::move(other.string_));
      break;
    case Tag::Tuple:
      new (&tuple_) Tuple(std::move(other.tuple_));
      break;
    case Tag::None:
      break;
  }
  other.tag_ = Tag::None;
}

Value &Value::operator=(Value &&other) noexcept {
  if (this != &other) {
    Destroy();
    tag_ = other.tag_;
    switch (tag_) {
      case Tag::Tensor:
        new (&tensor_) Tensor(std::move(other.tensor_));
        break;
      case Tag::Double:
        double_ = other.double_;
        break;
      case Tag::Int:
        int_ = other.int_;
        break;
      case Tag::Bool:
        bool_ = other.bool_;
        break;
      case Tag::String:
        new (&string_) std::string(std::move(other.string_));
        break;
      case Tag::Tuple:
        new (&tuple_) Tuple(std::move(other.tuple_));
        break;
      case Tag::None:
        break;
    }
    other.tag_ = Tag::None;
  }
  return *this;
}

#define CHECK_TAG(expected)                       \
  if (tag_ != expected) {                         \
    throw std::runtime_error("Bad Value access"); \
  }

Tensor Value::ToTensor() const {
  CHECK_TAG(Tag::Tensor);
  return tensor_;
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
const std::string &Value::ToString() const {
  CHECK_TAG(Tag::String);
  return string_;
}
Tuple Value::ToTuple() const {
  CHECK_TAG(Tag::Tuple);
  return tuple_;
}

std::ostream &operator<<(std::ostream &os, const Value &value) {
  switch (value.GetTag()) {
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
    case Value::Tag::Tuple: {
      os << "Tuple(";
      auto tuple = value.ToTuple();
      if (tuple.Defined() && !tuple.GetElements().empty()) {
        const auto &elements = tuple.GetElements();
        for (size_t i = 0; i < elements.size(); ++i) {
          os << elements[i];
          if (i < elements.size() - 1) {
            os << ", ";
          }
        }
      }
      os << ")";
      break;
    }
  }
  return os;
}

}  // namespace ir
}  // namespace mrt
