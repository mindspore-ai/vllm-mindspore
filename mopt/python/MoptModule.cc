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
#include "mlir/IR/MLIRContext.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"

#include "mopt/Dialect/Mrt/MrtDialect.h"

PYBIND11_MODULE(_mopt, m) {
  m.doc() = "mopt bindings";

  m.def("register_dialect", [](py::object context) {
    // Convert Python MLIR context to C++ MLIR context
    mlir::MLIRContext *cppContext = unwrap(mlirPythonCapsuleToContext(context.ptr()));
    if (!cppContext) {
      throw std::runtime_error("Failed to unwrap MLIR context");
    }

    // Register the dialect directly with the C++ context
    cppContext->getOrLoadDialect<mrt::MrtDialect>();
  });
}
