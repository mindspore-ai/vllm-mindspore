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

#ifndef MOPT_CONVERSION_STABLEHLO_TO_MRT_TYPE_CONVERTER_H
#define MOPT_CONVERSION_STABLEHLO_TO_MRT_TYPE_CONVERTER_H

#include "mopt/Conversion/MrtTypeConverter.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mopt {

// TypeConverter for StableHLO to MRT conversion
class StablehloToMrtTypeConverter : public mlir::TypeConverter {
 public:
  explicit StablehloToMrtTypeConverter(mlir::MLIRContext *ctx) {
    addConversion([](mlir::Type type) { return type; });
    mrt::populateMrtTypeConversions(*this);
    mrt::populateMrtTypeMaterializations(*this);
  }
};

}  // namespace mopt

#endif  // MOPT_CONVERSION_STABLEHLO_TO_MRT_TYPE_CONVERTER_H
