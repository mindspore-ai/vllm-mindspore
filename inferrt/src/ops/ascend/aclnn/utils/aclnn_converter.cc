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

#include <map>

#include "ops/ascend/aclnn/utils/aclnn_converter.h"

namespace mrt {
namespace ops {
static const std::map<ir::DataType::Type, aclDataType> kDataTypeToAclDataTypeMap = {
  {ir::DataType::Type::Unknown, ACL_DT_UNDEFINED},
  {ir::DataType::Type::Float32, ACL_FLOAT},
  {ir::DataType::Type::Float64, ACL_DOUBLE},
  {ir::DataType::Type::Int8, ACL_INT8},
  {ir::DataType::Type::Int16, ACL_INT16},
  {ir::DataType::Type::Int32, ACL_INT32},
  {ir::DataType::Type::Int64, ACL_INT64},
  {ir::DataType::Type::UInt8, ACL_UINT8},
  {ir::DataType::Type::Bool, ACL_BOOL},
};

aclDataType Convert(ir::DataType::Type dtype) {
  auto iter = kDataTypeToAclDataTypeMap.find(dtype);
  if (iter == kDataTypeToAclDataTypeMap.end()) {
    LOG_EXCEPTION << "Invalid dtype: " << dtype;
  }
  auto ret = iter->second;
  if (ret == ACL_DT_UNDEFINED) {
    LOG_EXCEPTION << "Invalid dtype: " << dtype;
  }
  return ret;
}

}  // namespace ops
}  // namespace mrt
