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

namespace mrt {
namespace ir {

class Value;
using ValuePtr = IntrusivePtr<Value>;

/**
 * @brief A tuple of values.
 *
 * This class provides holds a vector of Value objects.
 */
class Tuple {
 public:
  /**
   * @brief Default constructor. Creates an empty Tuple.
   */
  Tuple() = default;

  /**
   * @brief Constructs a Tuple from a vector of Value objects.
   * @param elements The vector of elements to include in the tuple.
   */
  explicit Tuple(const std::vector<ValuePtr> &elements) : elements_(elements) {}
  explicit Tuple(std::vector<ValuePtr> &&elements) : elements_(std::move(elements)) {}

  /**
   * @brief Move Constructor.
   * @param other The Tuple to move from.
   */
  Tuple(Tuple &&other) noexcept : elements_(std::move(other.elements_)) {}

   /**
   * @brief Get the size of the tuple.
   * @return The number of elements in the tuple.
   */
  size_t Size() const { return elements_.size(); }

  /**
   * @brief Get iterators to the beginning and end of the tuple elements.
   * @return A pair of const iterators to the beginning and end.
   */
  auto begin() const { return elements_.begin(); }
  auto end() const { return elements_.end(); }

  /**
   * @brief Retrieves the raw pointer of an element by index.
   * @param index The index of the element to retrieve.
   * @return The element as Value*, or nullptr if the index is out of bounds.
   */
  Value *operator[](size_t index) const {
    if (index < elements_.size()) {
      return elements_[index].get();
    }
    return nullptr;
  }

 private:
  std::vector<ValuePtr> elements_;
};

/**
 * @brief A generic container for different types of values.
 *
 * This class can hold a variety of types, including tensors, scalars, strings,
 * and tuples. It uses a tagged union to store the data efficiently.
 */
class Value : public RefCounted {
 public:
  /**
   * @brief Default constructor. Creates a None value.
   */
  Value() : tag_(Tag::None) {}
  /**
   * @brief Constructs a Value from a Tensor by moving.
   * @param v The Tensor value.
   */
  Value(Tensor &&v);
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
   * @brief Constructs a Value from a std::string by moving.
   * @param v The string value.
   */
  Value(std::string &&v);
  /**
   * @brief Constructs a Value from a Tuple by moving.
   * @param v The Tuple value.
   */
  Value(Tuple &&v);

  /**
   * @brief Destructor.
   */
  ~Value();

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
  const Tensor *ToTensor() const;
  Tensor *ToTensor();
  double ToDouble() const;
  int64_t ToInt() const;
  bool ToBool() const;
  const std::string *ToString() const;
  std::string *ToString();
  const Tuple *ToTuple() const;
  Tuple *ToTuple();
  ///@}

  /**
   * @brief Overloads the output stream operator for ValuePtr.
   * @param os The output stream.
   * @param value The ValuePtr to output.
   * @return The output stream.
   */
  friend std::ostream &operator<<(std::ostream &os, const Value *value);

 private:
  /**
   * @brief Enumeration of the possible types a Value can hold.
   */
  enum class Tag { None, Tensor, Double, Int, Bool, String, Tuple };

  const Tag tag_;  ///< The tag indicating the type of the value.
  union {
    Tensor tensor_;
    double double_;
    int64_t int_;
    bool bool_;
    std::string string_;
    Tuple tuple_;
  };
};

std::ostream &operator<<(std::ostream &os, const ValuePtr &value);
std::ostream &operator<<(std::ostream &os, const Value *value);

}  // namespace ir
}  // namespace mrt

#endif  // __IR_VALUE_VALUE_H__
