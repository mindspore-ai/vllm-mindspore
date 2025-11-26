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

#ifndef MOPT_C_DIALECTS_H
#define MOPT_C_DIALECTS_H

#include "mlir-c/RegisterEverything.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Mrt, mrt);

// Mrt TensorType CAPI
MLIR_CAPI_EXPORTED bool mlirTypeIsAMrtTensorType(MlirType type);
MLIR_CAPI_EXPORTED MlirTypeID mlirMrtTensorTypeGetTypeID(void);
MLIR_CAPI_EXPORTED MlirType mlirMrtTensorTypeGetElementType(MlirType type);
MLIR_CAPI_EXPORTED MlirAttribute mlirMrtTensorTypeGetShape(MlirType type);
MLIR_CAPI_EXPORTED MlirAttribute mlirMrtTensorTypeGetDevice(MlirType type);

// Mrt DeviceAttr CAPI
MLIR_CAPI_EXPORTED bool mlirAttributeIsAMrtDeviceAttr(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirTypeID mlirMrtDeviceAttrGetTypeID(void);
MLIR_CAPI_EXPORTED MlirAttribute mlirMrtDeviceAttrGetDeviceType(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirMrtDeviceAttrGetIndex(MlirAttribute attr);

#ifdef __cplusplus
}
#endif

#endif  // MOPT_C_DIALECTS_H
