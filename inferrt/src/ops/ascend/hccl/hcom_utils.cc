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

#include "ops/ascend/hccl/hcom_utils.h"

#include <set>
#include <algorithm>
#include <memory>
#include <utility>

namespace mrt::ops {

inline int64_t LongMulWithOverflowCheck(int64_t a, int64_t b) {
  int64_t out = a * b;
  if (a != 0) {
    bool overflow = ((out / a) != b);
    if (overflow) {
      LOG_EXCEPTION << "Mul: a(" << a << ") * b(" << b << ") result is overflow";
    }
  }
  return out;
}

inline size_t SizetMulWithOverflowCheck(size_t a, size_t b) {
  size_t out = a * b;
  if (a != 0) {
    if ((out / a) != b) {
      LOG_EXCEPTION << "Mul: a(" << a << ") * b(" << b << ") result is overflow";
    }
  }
  return out;
}

inline size_t LongToSizeClipNeg(int64_t u) { return u < 0 ? 0 : static_cast<size_t>(u); }

::HcclDataType HcomUtil::ConvertHcclType(DataType type_id) {
  auto iter = kConstOpHcomDataTypeMap.find(type_id);
  if (iter == kConstOpHcomDataTypeMap.end()) {
    LOG_EXCEPTION << "HcomDataType can't support Current Ascend Data Type : " << type_id.ToString();
  }
  return iter->second;
}

bool HcomUtil::GetHcclOpSize(const HcclDataType &data_type, const std::vector<int64_t> &shape, size_t *size) {
  CHECK_IF_NULL(size);
  int64_t tmp_size = 1;
  uint32_t type_size = 4;
  for (size_t i = 0; i < shape.size(); i++) {
    tmp_size = LongMulWithOverflowCheck(tmp_size, shape[i]);
  }

  if (!GetHcomTypeSize(data_type, &type_size)) {
    return false;
  }

  *size = SizetMulWithOverflowCheck(LongToSizeClipNeg(tmp_size), type_size);
  return true;
}

bool HcomUtil::GetHcomTypeSize(const HcclDataType &data_type, uint32_t *size) {
  CHECK_IF_NULL(size);
  auto iter = kConstOpHcomDataTypeSizeMap.find(data_type);
  if (iter == kConstOpHcomDataTypeSizeMap.end()) {
    LOG_ERROR << "HcomUtil::HcomDataTypeSize, No DataTypeSize!";
    return false;
  }
  *size = iter->second;
  return true;
}

bool HcomUtil::GetHcomCount(const std::vector<HcclDataType> &data_type_list,
                            const std::vector<std::vector<int64_t>> &shape_list, const size_t input_tensor_num,
                            const std::optional<int64_t> rank_size_opt, uint64_t *total_count) {
  CHECK_IF_NULL(total_count);

  const uint32_t align_size = 512;
  const uint32_t filled_size = 32;
  uint64_t total_size = 0;
  size_t input_size;
  uint32_t type_size = 4;
  // size_t rank_size = 1;
  CHECK_IF_FAIL(data_type_list.size() == shape_list.size());

  for (size_t i = 0; i < data_type_list.size(); ++i) {
    if (!GetHcomTypeSize(data_type_list[i], &type_size)) {
      return false;
    }

    if (!GetHcclOpSize(data_type_list[i], shape_list[i], &input_size)) {
      LOG_ERROR << "Get GetHcclOpSize failed";
      return false;
    }

    if (input_tensor_num > 1) {
      // communication operator with dynamic input should have continuous memory.
      input_size = (input_size + align_size - 1 + filled_size) / align_size * align_size;
    }

    bool all_dynamic = std::all_of(shape_list[i].begin(), shape_list[i].end(), [](int64_t x) { return x == -1; });
    if (!all_dynamic && (type_size == 0 || input_size % type_size != 0)) {
      return false;
    }
    total_size += input_size / type_size;
  }
  *total_count = total_size;
  return true;
}

std::pair<uint64_t, ::HcclDataType> HcomUtil::GetHcclCountAndTypeFromTensor(
  const ir::TensorPtr &tensor, const std::optional<int64_t> rank_size_opt) {
  auto type_id = tensor->Dtype();
  auto shape = tensor->Shape();

  auto hccl_type = ConvertHcclType(type_id);

  uint64_t hccl_count = 0;
  constexpr size_t input_tensor_size = 1;
  if (!GetHcomCount({hccl_type}, {shape}, input_tensor_size, rank_size_opt, &hccl_count)) {
    LOG_EXCEPTION << "GetHcomCount fail!";
  }
  return std::make_pair(hccl_count, hccl_type);
}

CollectiveOpReduceType HcomUtil::GetCollectiveOpReduceType(const std::string &reduce_op) {
  auto iter = kConstOpCollectiveOpReduceTypeMap.find(reduce_op);
  if (iter == kConstOpCollectiveOpReduceTypeMap.end()) {
    LOG_EXCEPTION << "HcomUtil::Get CollectiveOpReduceType fail, [" << reduce_op << "] not support!";
  }
  return iter->second;
}

HcclReduceOp HcomUtil::GetHcomReduceOpType(const std::string &reduce_op) {
  auto iter = kConstOpHcomReduceOpTypeMap.find(reduce_op);
  if (iter == kConstOpHcomReduceOpTypeMap.end()) {
    LOG_EXCEPTION << "HcomUtil::Get HCOM_ATTR_REDUCE_TYPE fail, [" << reduce_op << "] not support!";
  }
  return iter->second;
}

}  // namespace mrt::ops
