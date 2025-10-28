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

#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"

#include "mopt-c/Registration.h"
#include "mopt/Dialect/Mrt/MrtDialect.h"

#ifdef __cplusplus
extern "C" {
#endif

void moptRegisterAllDialects(MlirContext context) {
  mlir::DialectRegistry registry;
  registry.insert<mrt::MrtDialect>();
  unwrap(context)->appendDialectRegistry(registry);
  unwrap(context)->loadAllAvailableDialects();
}

void moptRegisterAllPasses(void) {
  // Register passes when they are implemented
}

#ifdef __cplusplus
}
#endif
