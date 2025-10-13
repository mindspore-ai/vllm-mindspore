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

#ifndef OPS_ASCEND_HCCL_HCOM_UTILS_H_
#define OPS_ASCEND_HCCL_HCOM_UTILS_H_

#include <string>
#include <map>
#include <vector>
#include <memory>
#include <unordered_map>
#include <utility>
#include <optional>

#include "hardware/hardware_abstract/collective/collective_manager.h"
#include "ir/common/dtype.h"
#include "ir/tensor/tensor.h"
#include "common/common.h"

#include "hccl/base.h"
#include "hccl/hccl_types.h"

namespace mrt::ops {
using ir::DataType;
using mrt::ir::TensorPtr;
using std::map;
using std::string;
using std::vector;
constexpr int64_t kComplex64ConvertFloat32Num = 2;

enum CollectiveOpReduceType : int64_t {
  Reduce_Mean = 0,
  Reduce_Max = 1,
  Reduce_Min = 2,
  Reduce_Prod = 3,
  Reduce_Sum = 4,
  Reduce_Sum_Square = 5,
  Reduce_ASum = 6,
  Reduce_All = 7
};

/* Correspondence between data_type and hcom data type in Ascend */
static const map<int64_t, HcclDataType> kConstOpHcomDataTypeMap = {
  {DataType::Int8, HCCL_DATA_TYPE_INT8},
  {DataType::Int16, HCCL_DATA_TYPE_INT16},
  {DataType::Int32, HCCL_DATA_TYPE_INT32},
  {DataType::Float32, HCCL_DATA_TYPE_FP32},
  {DataType::Int64, HCCL_DATA_TYPE_INT64},
  {DataType::UInt8, HCCL_DATA_TYPE_UINT8},
  {DataType::Float64, HCCL_DATA_TYPE_FP64},
  {DataType::Bool, HCCL_DATA_TYPE_INT8},
#ifdef EXPERIMENT_A5
  {DataType::kNumberTypeHiFloat8, HCCL_DATA_TYPE_HIF8},
  {DataType::kNumberTypeFloat8E5M2, HCCL_DATA_TYPE_FP8E5M2},
  {DataType::kNumberTypeFloat8E4M3FN, HCCL_DATA_TYPE_FP8E4M3},
#endif
};

/* Correspondence between data_type and occupied byte size in hcom */
static const map<HcclDataType, uint32_t> kConstOpHcomDataTypeSizeMap = {
  {HCCL_DATA_TYPE_INT8, sizeof(int8_t)},         {HCCL_DATA_TYPE_INT16, sizeof(int32_t) / 2},
  {HCCL_DATA_TYPE_INT32, sizeof(int32_t)},       {HCCL_DATA_TYPE_FP16, sizeof(float) / 2},
  {HCCL_DATA_TYPE_FP32, sizeof(float)},          {HCCL_DATA_TYPE_INT64, sizeof(int64_t)},
  {HCCL_DATA_TYPE_UINT64, sizeof(uint64_t)},     {HCCL_DATA_TYPE_UINT8, sizeof(uint8_t)},
  {HCCL_DATA_TYPE_UINT16, sizeof(uint32_t) / 2}, {HCCL_DATA_TYPE_UINT32, sizeof(uint32_t)},
  {HCCL_DATA_TYPE_FP64, sizeof(double)},         {HCCL_DATA_TYPE_BFP16, sizeof(float) / 2},
#ifdef EXPERIMENT_A5
  {HCCL_DATA_TYPE_HIF8, sizeof(float) / 4},      {HCCL_DATA_TYPE_FP8E5M2, sizeof(float) / 4},
  {HCCL_DATA_TYPE_FP8E4M3, sizeof(float) / 4},
#endif
};

static const std::map<CollectiveOpReduceType, HcclReduceOp> kHcomOpReduceTypeMap = {
  {CollectiveOpReduceType::Reduce_Max, HCCL_REDUCE_MAX},
  {CollectiveOpReduceType::Reduce_Min, HCCL_REDUCE_MIN},
  {CollectiveOpReduceType::Reduce_Prod, HCCL_REDUCE_PROD},
  {CollectiveOpReduceType::Reduce_Sum, HCCL_REDUCE_SUM}};

/* Correspondence between reduce str and enum in hcom  */
static const std::unordered_map<std::string, HcclReduceOp> kConstOpHcomReduceOpTypeMap = {
  {"min", HCCL_REDUCE_MIN},
  {"max", HCCL_REDUCE_MAX},
  {"prod", HCCL_REDUCE_PROD},
  {"sum", HCCL_REDUCE_SUM},
};

/* Correspondence between reduce str and enum in collective op  */
static const std::unordered_map<std::string, CollectiveOpReduceType> kConstOpCollectiveOpReduceTypeMap = {
  {"min", CollectiveOpReduceType::Reduce_Min},
  {"max", CollectiveOpReduceType::Reduce_Max},
  {"prod", CollectiveOpReduceType::Reduce_Prod},
  {"sum", CollectiveOpReduceType::Reduce_Sum},
};

class HcomUtil {
 public:
  static ::HcclDataType ConvertHcclType(DataType type_id);
  static HcclComm LoadHcclLibrary(const std::string &group_name) {
    int64_t hccl_comm = collective::CollectiveManager::Instance().GetCommunicationGroup(group_name)->communicator();
    return reinterpret_cast<HcclComm>(static_cast<intptr_t>(hccl_comm));
  }
  // static bool GetHcomDataType(const std::string &kernel_name, const std::vector<TensorPtr> &inputs,
  //                             const std::vector<TensorPtr> &outputs, std::vector<HcclDataType> *data_type_list);
  static bool GetHcclOpSize(const HcclDataType &data_type, const std::vector<int64_t> &shape, size_t *size);
  static bool GetHcomTypeSize(const HcclDataType &data_type, uint32_t *size);
  static bool GetHcomCount(const std::vector<HcclDataType> &data_type_list,
                           const std::vector<std::vector<int64_t>> &shape_list, const size_t input_tensor_num,
                           const std::optional<int64_t> rank_size_opt, uint64_t *total_count);

  static std::pair<uint64_t, ::HcclDataType> GetHcclCountAndTypeFromTensor(
    const ir::TensorPtr &tensor, const std::optional<int64_t> rank_size_opt = std::nullopt);
  static CollectiveOpReduceType GetCollectiveOpReduceType(const std::string &reduce_op);
  static HcclReduceOp GetHcomReduceOpType(const std::string &reduce_op);
};
}  // namespace mrt::ops

#endif  // OPS_ASCEND_HCCL_HCOM_UTILS_H_
