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

#ifndef MRT_DIALECT_MRT_OPS_H
#define MRT_DIALECT_MRT_OPS_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"

#include "mopt/Dialect/Mrt/MrtDialect.h"

#define GET_ATTRDEF_CLASSES
#include "mopt/Dialect/Mrt/MrtAttributes.h.inc"

#define GET_TYPEDEF_CLASSES
#include "mopt/Dialect/Mrt/MrtTypes.h.inc"

#define GET_OP_CLASSES
#include "mopt/Dialect/Mrt/Mrt.h.inc"

#endif  // MRT_DIALECT_MRT_OPS_H
