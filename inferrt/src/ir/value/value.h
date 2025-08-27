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

#ifndef __IR_VALUE_VALUE_H__
#define __IR_VALUE_VALUE_H__

#include <vector>
#include <string>
#include <memory>
#include <stdexcept>

#include "ir/common/intrusive_ptr.h"
#include "ir/tensor/tensor.h"
#include "ir/value/tuple.h"

namespace mrt {
namespace ir {

/**
 * @brief A generic container for different types of values.
 *
 * This class can hold a variety of types, including tensors, scalars, strings,
 * and tuples. It uses a tagged union to store the data efficiently.
 */
class Value {
 public:
  /**
   * @brief Enumeration of the possible types a Value can hold.
   */
  enum class Tag { None, Tensor, Double, Int, Bool, String, Tuple };

  /**
   * @brief Default constructor. Creates a None value.
   */
  Value() : tag_(Tag::None) {}
  /**
   * @brief Constructs a Value from a Tensor.
   * @param v The Tensor value.
   */
  Value(Tensor v);
  /**
   * @brief Constructs a Value from a double.
   * @param v The double value.
   */
  Value(double v);
  /**
   * @brief Constructs a Value from an int64_t.
   * @param v The int64_t value.
   */
  Value(int64_t v);
  /**
   * @brief Constructs a Value from a bool.
   * @param v The bool value.
   */
  Value(bool v);
  /**
   * @brief Constructs a Value from a std::string.
   * @param v The string value.
   */
  Value(std::string v);
  /**
   * @brief Constructs a Value from a Tuple.
   * @param v The Tuple value.
   */
  Value(Tuple v);

  /**
   * @brief Move constructor.
   * @param other The Value to move from.
   */
  Value(Value &&other) noexcept;
  /**
   * @brief Move assignment operator.
   * @param other The Value to move from.
   * @return *this
   */
  Value &operator=(Value &&other) noexcept;

  /**
   * @brief Copy constructor.
   * @param other The Value to copy from.
   */
  Value(const Value &other);
  /**
   * @brief Copy assignment operator.
   * @param other The Value to copy from.
   * @return *this
   */
  Value &operator=(const Value &other);

  /**
   * @brief Destructor.
   */
  ~Value();

  /**
   * @brief Get the tag of the value.
   * @return The tag.
   */
  Tag GetTag() const { return tag_; }

  /**
   * @brief Set the tag of the value.
   * @param tag The tag.
   */
  void SetTag(Tag tag) { tag_ = tag; }

  /** @name Type checkers */
  ///@{
  bool IsTensor() const { return tag_ == Tag::Tensor; }
  bool IsDouble() const { return tag_ == Tag::Double; }
  bool IsInt() const { return tag_ == Tag::Int; }
  bool IsBool() const { return tag_ == Tag::Bool; }
  bool IsString() const { return tag_ == Tag::String; }
  bool IsTuple() const { return tag_ == Tag::Tuple; }
  bool IsNone() const { return tag_ == Tag::None; }
  ///@}

  /** @name Value extractors
   *  These methods extract the underlying value. They will throw a
   *  std::runtime_error if the type does not match.
   */
  ///@{
  Tensor ToTensor() const;
  double ToDouble() const;
  int64_t ToInt() const;
  bool ToBool() const;
  const std::string &ToString() const;
  Tuple ToTuple() const;
  ///@}

 private:
  /**
   * @brief Destroys the currently held value.
   */
  void Destroy();
  /**
   * @brief Copies the value from another Value object.
   * @param other The Value to copy from.
   */
  void CopyFrom(const Value &other);

  Tag tag_;  ///< The tag indicating the type of the value.
  union {
    Tensor tensor_;
    double double_;
    int64_t int_;
    bool bool_;
    std::string string_;
    Tuple tuple_;
  };
};

std::ostream &operator<<(std::ostream &os, const Value &value);

}  // namespace ir
}  // namespace mrt

#endif  // __IR_VALUE_VALUE_H__
