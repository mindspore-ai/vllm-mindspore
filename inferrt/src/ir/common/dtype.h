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

#ifndef __IR_COMMON_DTYPE_H__
#define __IR_COMMON_DTYPE_H__

#include <cstdint>
#include <string>
#include <stdexcept>

namespace mrt {
namespace ir {

/**
 * @brief Represents a data type.
 */
struct DataType {
  /**
   * @brief Enumeration of supported data types.
   */
  enum Type : int8_t {
    Unknown,  ///< Unknown data type
    Float32,  ///< 32-bit floating point
    Float64,  ///< 64-bit floating point
    Int8,     ///< 8-bit signed integer
    Int16,    ///< 16-bit signed integer
    Int32,    ///< 32-bit signed integer
    Int64,    ///< 64-bit signed integer
    UInt8,    ///< 8-bit unsigned integer
    Bool,     ///< Boolean
  };

  Type value;  ///< The underlying enum value.

  /**
   * @brief Default constructor.
   */
  DataType() : value(Unknown) {}

  /**
   * @brief Constructs a DataType from a Type enum.
   * @param v The enum value.
   */
  DataType(Type v) : value(v) {}

  /**
   * @brief Allows for using DataType in a switch statement.
   */
  operator Type() const { return value; }

  /**
   * @brief Gets the size of a data type in bytes.
   * @return The size in bytes.
   * @throws std::runtime_error if the data type is unsupported.
   */
  size_t GetSize() const {
    switch (value) {
      case Float32:
        return 4;
      case Float64:
        return 8;
      case Int8:
        return 1;
      case Int16:
        return 2;
      case Int32:
        return 4;
      case Int64:
        return 8;
      case UInt8:
        return 1;
      case Bool:
        return 1;
      default:
        throw std::runtime_error("Unsupported data type");
    }
  }

  /**
   * @brief Converts a data type to its string representation.
   * @return The string representation.
   * @throws std::runtime_error if the data type is unsupported.
   */
  std::string ToString() const {
    switch (value) {
      case Float32:
        return "float32";
      case Float64:
        return "float64";
      case Int8:
        return "int8";
      case Int16:
        return "int16";
      case Int32:
        return "int32";
      case Int64:
        return "int64";
      case UInt8:
        return "uint8";
      case Bool:
        return "bool";
      case Unknown:
        return "unknown";
      default:
        throw std::runtime_error("Unsupported data type");
    }
  }
};

}  // namespace ir
}  // namespace mrt

#endif  // __IR_COMMON_DTYPE_H__
