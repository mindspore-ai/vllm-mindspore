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

#include "mopt-c/Dialects.h"
#include "mopt-c/Passes.h"
#include "mlir-c/IR.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "mlir/CAPI/IR.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir-c/BuiltinAttributes.h"

namespace py = pybind11;
using mlir::python::adaptors::mlir_attribute_subclass;
using mlir::python::adaptors::mlir_type_subclass;

PYBIND11_MODULE(_mopt, m) {
  mlirRegisterMoptPasses();

  m.doc() = "mopt python bindings";

  m.def(
    "register_mrt_dialect",
    [](MlirContext context, bool load) {
      MlirDialectHandle handle = mlirGetDialectHandle__mrt__();
      mlirDialectHandleRegisterDialect(handle, context);
      if (load) {
        mlirDialectHandleLoadDialect(handle, context);
      }
    },
    py::arg("context"), py::arg("load") = true, "Register MRT dialect");

  m.def(
    "register_dvm_dialect",
    [](MlirContext context, bool load) {
      MlirDialectHandle handle = mlirGetDialectHandle__dvm__();
      mlirDialectHandleRegisterDialect(handle, context);
      if (load) {
        mlirDialectHandleLoadDialect(handle, context);
      }
    },
    py::arg("context"), py::arg("load") = true, "Register DVM dialect");

  // Bind Mrt::DeviceAttr through CAPI
  auto deviceAttrClass =
    mlir_attribute_subclass(m, "DeviceAttr", mlirAttributeIsAMrtDeviceAttr, mlirMrtDeviceAttrGetTypeID)
      .def_property_readonly(
        "device_type",
        [](MlirAttribute self) -> std::string {
          MlirAttribute typeAttr = mlirMrtDeviceAttrGetDeviceType(self);
          MlirStringRef typeStr = mlirStringAttrGetValue(typeAttr);
          return std::string(typeStr.data, typeStr.length);
        },
        "Get the device type (e.g., 'cpu', 'npu')")
      .def_property_readonly(
        "index",
        [](MlirAttribute self) -> int64_t {
          MlirAttribute indexAttr = mlirMrtDeviceAttrGetIndex(self);
          return mlirIntegerAttrGetValueInt(indexAttr);
        },
        "Get the device index");

  // Bind Mrt::TensorType through CAPI
  // Expose shape and element_type as properties (like RankedTensorType)
  mlir_type_subclass(m, "TensorType", mlirTypeIsAMrtTensorType, mlirMrtTensorTypeGetTypeID)
    .def_property_readonly(
      "element_type", [](MlirType self) { return mlirMrtTensorTypeGetElementType(self); },
      "Get the element type of the tensor")
    .def_property_readonly(
      "shape", [](MlirType self) { return mlirMrtTensorTypeGetShape(self); }, "Get the shape of the tensor")
    .def_property_readonly(
      "device",
      [](MlirType self) -> py::object {
        MlirAttribute deviceAttr = mlirMrtTensorTypeGetDevice(self);
        if (!mlirAttributeIsNull(deviceAttr)) {
          return py::cast(deviceAttr);
        }
        return py::none();
      },
      "Get the device attribute of the tensor");
}
