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

#include "include/custom_op_api.h"

namespace mrt {
namespace ops {
class CpuCustomAddOperator : public CPUCustomOperator {
 public:
  CpuCustomAddOperator() = default;
  virtual ~CpuCustomAddOperator() = default;

  OpsErrorCode CalcWorkspace(const std::vector<const ir::Value *> &input, const ir::Value *output,
                             size_t *workspaceSize) override {
    return SUCCESS;
  }

  // Template function to perform element-wise addition
  template <typename T>
  OpsErrorCode AddImpl(const void *input1_data, const void *input2_data, void *output_data, size_t tensor_size) {
    const T *in1 = static_cast<const T *>(input1_data);
    const T *in2 = static_cast<const T *>(input2_data);
    T *out = static_cast<T *>(output_data);
    
    // Perform element-wise addition
    for (size_t i = 0; i < tensor_size; ++i) {
      out[i] = in1[i] + in2[i];
    }
    
    return SUCCESS;
  }

  OpsErrorCode Launch(const std::vector<const ir::Value *> &input, void *workspace, size_t workspaceSize,
                      ir::Value *output, void *stream) override {
    // Check input parameters
    if (input.size() != 2) {
      LOG_ERROR << "CpuCustomAddOp expects 2 inputs, got " << input.size();
      return INVALID_INPUT_NUM;
    }
    
    // Get input and output tensors
    auto input1 = input[0]->ToTensor();
    auto input2 = input[1]->ToTensor();
    auto output_tensor = output->ToTensor();
    
    if (!input1 || !input2 || !output_tensor) {
      LOG_ERROR << "Failed to convert values to tensors";
      return INVALID_PARAM;
    }
    
    // Get tensor data and size
    const void *input1_data = input1->DataPtr();
    const void *input2_data = input2->DataPtr();
    void *output_data = output_tensor->DataPtr();
    
    if (!input1_data || !input2_data || !output_data) {
      LOG_ERROR << "Null tensor data pointer";
      return INVALID_PARAM;
    }
    
    // Get tensor shape
    auto tensor_shape = input1->Shape();
    if (tensor_shape != input2->Shape() || tensor_shape != output_tensor->Shape()) {
      LOG_ERROR << "Tensor shape mismatch";
      return INVALID_PARAM;
    }
    
    // Perform element-wise addition based on data type
    ir::DataType dtype = input1->Dtype();
    if (dtype != input2->Dtype() || dtype != output_tensor->Dtype()) {
      LOG_ERROR << "Tensor dtype mismatch";
      return INVALID_PARAM;
    }
    
    auto tensor_size = input1->Numel();
    switch (dtype) {
      case ir::DataType::Float32:
        return AddImpl<float>(input1_data, input2_data, output_data, tensor_size);
      case ir::DataType::Float64:
        return AddImpl<double>(input1_data, input2_data, output_data, tensor_size);
      case ir::DataType::Int32:
        return AddImpl<int32_t>(input1_data, input2_data, output_data, tensor_size);
      case ir::DataType::Int64:
        return AddImpl<int64_t>(input1_data, input2_data, output_data, tensor_size);
      default:
        LOG_ERROR << "Unsupported data type for CpuCustomAddOp: " << dtype.ToString();
        return INVALID_PARAM;
    }
    return SUCCESS;
  }
};

REGISTER_CUSTOM_OP(custom_add, CpuCustomAddOperator);
}  // namespace ops
}  // namespace mrt
