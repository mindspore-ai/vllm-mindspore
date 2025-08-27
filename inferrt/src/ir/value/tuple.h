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

#ifndef __IR_VALUE_TUPLE_H__
#define __IR_VALUE_TUPLE_H__

#include <vector>

#include "ir/common/intrusive_ptr.h"

namespace mrt {
namespace ir {

class Value;

/**
 * @brief Implementation of a tuple of values.
 *
 * This struct is reference-counted and holds a vector of Value objects.
 */
struct TupleImpl : public RefCounted {
  explicit TupleImpl(const std::vector<Value> &elements) : elements(elements) {}
  explicit TupleImpl(std::vector<Value> &&elements) : elements(std::move(elements)) {}
  std::vector<Value> elements;  ///< The elements of the tuple.
};

/**
 * @brief Representing a tuple of values.
 *
 * This class provides a user-friendly interface to the tuple implementation,
 * managing the lifetime of the underlying TupleImpl using an IntrusivePtr.
 */
class Tuple {
 public:
  /**
   * @brief Default constructor. Creates an empty Tuple.
   */
  Tuple();

  /**
   * @brief Constructs a Tuple from a vector of Value objects.
   * @param elements The vector of elements to include in the tuple.
   */
  explicit Tuple(const std::vector<Value> &elements);

  /**
   * @brief Move Constructor from a vector of Value objects.
   * @param elements The vector of elements to include in the tuple.
   */
  explicit Tuple(const std::vector<Value> &&elements);

  /**
   * @brief Destructor.
   */
  ~Tuple();

  /**
   * @brief Copy constructor.
   * @param other The Tuple to copy from.
   */
  Tuple(const Tuple &other);

  /**
   * @brief Move constructor.
   * @param other The Tuple to move from.
   */
  Tuple(Tuple &&other) noexcept;

  /**
   * @brief Copy assignment operator.
   * @param other The Tuple to copy from.
   * @return *this
   */
  Tuple &operator=(const Tuple &other);

  /**
   * @brief Move assignment operator.
   * @param other The Tuple to move from.
   * @return *this
   */
  Tuple &operator=(Tuple &&other) noexcept;

  /**
   * @brief Gets the elements of the tuple.
   * @return A const reference to the vector of elements.
   */
  const std::vector<Value> &GetElements() const;

  /**
   * @brief Checks if the tuple is defined (not null).
   * @return true if the tuple is defined, false otherwise.
   */
  bool Defined() const;

 private:
  IntrusivePtr<TupleImpl> impl_;
};

}  // namespace ir
}  // namespace mrt

#endif  // __IR_VALUE_TUPLE_H__
