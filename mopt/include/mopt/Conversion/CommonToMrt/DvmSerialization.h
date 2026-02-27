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

#ifndef MOPT_CONVERSION_COMMON_TO_MRT_DVM_SERIALIZATION_H
#define MOPT_CONVERSION_COMMON_TO_MRT_DVM_SERIALIZATION_H

#include <string>
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace dvm {

/**
 * @brief Serializes a DVM function (containing DVM dialect operations) to a JSON string.
 *
 * @param funcOp The function to serialize.
 * @param kernelType The type of kernel (e.g., "dyn_shape", "static_shape").
 * @param indent JSON indentation level.
 * @param jsonOut Output string for the JSON payload.
 * @param callInputsOut Output vector for the values that should be passed as inputs to the dvm_call.
 * @return mlir::LogicalResult Success or failure.
 */
mlir::LogicalResult serializeDvmFuncToJson(mlir::func::FuncOp funcOp, llvm::StringRef kernelType, unsigned indent,
                                           std::string &jsonOut, llvm::SmallVectorImpl<mlir::Value> &callInputsOut);

}  // namespace dvm
}  // namespace mlir

#endif  // MOPT_CONVERSION_COMMON_TO_MRT_DVM_SERIALIZATION_H
