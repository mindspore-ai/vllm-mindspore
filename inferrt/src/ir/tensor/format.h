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

#ifndef __IR_TENSOR_FORMAT_H__
#define __IR_TENSOR_FORMAT_H__

#include <cstdint>
#include <vector>
#include <map>
#include <string>

namespace mrt {
namespace ir {
enum MemoryFormat : int8_t {
  DEFAULT_FORMAT = -1,
  ND,  // Nd Tensor
  FRACTAL_NZ,
  NC1HWC0,    // NC1HWC0
  FRACTAL_Z,  // FRACTAL_Z
  NUM_OF_FORMAT
};

inline const std::vector<std::string> &GetFormatNames() {
  static std::vector<std::string> names = {
    "ND",
    "FRACTAL_NZ",
    "NC1HWC0",
    "FRACTAL_Z",
  };
  return names;
}

inline const std::map<std::string, MemoryFormat> &GetFormatStrToEnumMap() {
  static std::map<std::string, MemoryFormat> formatStrToEnumMap = {
    {"DefaultFormat", MemoryFormat::DEFAULT_FORMAT}, {"ND", MemoryFormat::ND},
    {"FRACTAL_NZ", MemoryFormat::FRACTAL_NZ},        {"NC1HWC0", MemoryFormat::NC1HWC0},
    {"FRACTAL_Z", MemoryFormat::FRACTAL_Z},
  };
  return formatStrToEnumMap;
}

inline std::string FormatEnumToString(MemoryFormat format) {
  const auto &names = GetFormatNames();
  if (format == MemoryFormat::DEFAULT_FORMAT) {
    return "DefaultFormat";
  }
  if (format < MemoryFormat::ND || format >= MemoryFormat::NUM_OF_FORMAT) {
    return "";
  }
  return names[format];
}

inline MemoryFormat FormatFromStrToEnum(const std::string &formatStr) {
  const auto &formatStrToEnumMap = GetFormatStrToEnumMap();
  auto it = formatStrToEnumMap.find(formatStr);
  if (it != formatStrToEnumMap.end()) {
    return it->second;
  }
  return MemoryFormat::DEFAULT_FORMAT;
}

}  // namespace ir
}  // namespace mrt
#endif  // __IR_TENSOR_FORMAT_H__
