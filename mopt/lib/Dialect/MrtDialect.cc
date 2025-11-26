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

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/SMLoc.h"

#include "mopt/Dialect/Mrt/Mrt.h"
#include "mopt/Dialect/Mrt/MrtDialect.h"

#include "mopt/Dialect/Mrt/MrtDialect.cpp.inc"

#define GET_OP_CLASSES
#include "mopt/Dialect/Mrt/Mrt.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "mopt/Dialect/Mrt/MrtTypes.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "mopt/Dialect/Mrt/MrtAttributes.cpp.inc"

// Custom parser and printer for Mrt_DeviceAttr
// Format: <"type", index>
// Example: <"npu", 0>
namespace mrt {
mlir::Attribute DeviceAttr::parse(mlir::AsmParser &parser, mlir::Type type) {
  std::string typeStr;
  int64_t index;

  if (parser.parseLess()) {
    return {};
  }

  // Parse string type
  if (parser.parseString(&typeStr)) {
    return {};
  }

  if (parser.parseComma()) {
    return {};
  }

  // Parse integer index (without type suffix)
  if (parser.parseInteger(index)) {
    return {};
  }

  if (parser.parseGreater()) {
    return {};
  }

  auto *ctx = parser.getContext();
  auto typeAttr = mlir::StringAttr::get(ctx, typeStr);
  auto indexAttr = mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64), index);
  return DeviceAttr::get(ctx, typeAttr, indexAttr);
}

void DeviceAttr::print(mlir::AsmPrinter &printer) const {
  printer << "<";
  printer << "\"" << getDeviceType().getValue() << "\"";
  printer << ", ";
  printer << getIndex().getValue().getSExtValue();
  printer << ">";
}

// Custom parser and printer for Mrt_TensorType
// Format: mrt.tensor<elementTypexshape, device=...>
// Example: mrt.tensor<f32x2x3, device=<npu, 0>>
mlir::Type TensorType::parse(mlir::AsmParser &odsParser) {
  mlir::Type elementType;
  mlir::DenseI64ArrayAttr shape;
  mrt::DeviceAttr device;

  if (odsParser.parseLess()) {
    return {};
  }

  // Parse elementTypexshape (e.g., f32x2x3)
  if (odsParser.parseType(elementType)) {
    return {};
  }

  // Parse shape dimensions separated by 'x'
  llvm::SmallVector<int64_t> shapeVec;
  while (odsParser.parseOptionalKeyword("x").succeeded()) {
    int64_t dim;
    if (odsParser.parseInteger(dim)) {
      return {};
    }
    shapeVec.push_back(dim);
  }

  // Create shape attribute
  shape = mlir::DenseI64ArrayAttr::get(odsParser.getContext(), shapeVec);

  // Parse optional device
  if (odsParser.parseOptionalComma().succeeded()) {
    if (odsParser.parseKeyword("device") || odsParser.parseEqual()) {
      return {};
    }
    mlir::Attribute deviceAttr;
    if (odsParser.parseAttribute(deviceAttr)) {
      return {};
    }
    device = mlir::dyn_cast<mrt::DeviceAttr>(deviceAttr);
    if (!device) {
      odsParser.emitError(odsParser.getNameLoc(), "expected DeviceAttr");
      return {};
    }
  }

  if (odsParser.parseGreater()) {
    return {};
  }

  return mrt::TensorType::get(odsParser.getContext(), elementType, shape, device);
}

void TensorType::print(mlir::AsmPrinter &odsPrinter) const {
  odsPrinter << "<";
  odsPrinter.printType(getElementType());

  // Print shape as x2x3 format (shape is required, so always print)
  auto shape = getShape();
  auto shapeArray = shape.asArrayRef();
  for (int64_t dim : shapeArray) {
    odsPrinter << "x" << dim;
  }

  // Print optional device
  auto device = getDevice();
  if (device) {
    odsPrinter << ", device=";
    device.print(odsPrinter);
  }

  odsPrinter << ">";
}
}  // namespace mrt

void mrt::MrtDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mopt/Dialect/Mrt/Mrt.cpp.inc"  // NOLINT(build/include)
    >();

  addTypes<
#define GET_TYPEDEF_LIST
#include "mopt/Dialect/Mrt/MrtTypes.cpp.inc"  // NOLINT(build/include)
    >();

  addAttributes<
#define GET_ATTRDEF_LIST
#include "mopt/Dialect/Mrt/MrtAttributes.cpp.inc"  // NOLINT(build/include)
    >();
}
