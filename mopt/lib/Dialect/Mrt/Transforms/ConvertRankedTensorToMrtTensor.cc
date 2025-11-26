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

#include "mopt/Dialect/Mrt/Transforms/Passes.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/SmallVector.h"
#include "mopt/Dialect/Mrt/Mrt.h"
#include "mopt/Dialect/Mrt/MrtDialect.h"

using mlir::ArrayRef;
using mlir::cast;
using mlir::dyn_cast;
using mlir::isa;
using mlir::MLIRContext;
using mlir::ModuleOp;
using mlir::Operation;
using mlir::OperationPass;
using mlir::Pass;
using mlir::PassWrapper;
using mlir::RankedTensorType;
using mlir::SmallVector;
using mlir::Type;

namespace {

// Convert RankedTensorType to Mrt_TensorType (device is optional)
Type convertToMrtTensorType(MLIRContext *ctx, Type type) {
  if (auto rankedType = dyn_cast<RankedTensorType>(type)) {
    mlir::Type elementType = rankedType.getElementType();
    auto shapeVec = llvm::to_vector(rankedType.getShape());
    mlir::DenseI64ArrayAttr shape = mlir::DenseI64ArrayAttr::get(ctx, shapeVec);
    // device is optional, can be nullptr
    return mrt::TensorType::get(ctx, elementType, shape, nullptr);
  }
  // If already Mrt_TensorType, return as is
  if (isa<mrt::TensorType>(type)) {
    return type;
  }
  return type;  // fallback for other types
}

// Convert all RankedTensorType to Mrt_TensorType in the module
void convertAllTypesToMrtTensorType(ModuleOp module) {
  MLIRContext *ctx = module.getContext();
  module.walk([&](Operation *op) {
    // Skip function operations as they will be updated separately
    if (isa<mlir::func::FuncOp>(op)) {
      return;
    }

    // Convert result types
    for (auto result : op->getResults()) {
      Type oldType = result.getType();
      if (auto rankedType = dyn_cast<RankedTensorType>(oldType)) {
        Type newType = convertToMrtTensorType(ctx, rankedType);
        if (newType != oldType) {
          result.setType(newType);
        }
      }
    }
  });

  // Convert function argument and result types
  module.walk([&](mlir::func::FuncOp funcOp) {
    // Convert argument types
    for (auto arg : funcOp.getArguments()) {
      Type oldType = arg.getType();
      if (auto rankedType = dyn_cast<RankedTensorType>(oldType)) {
        Type newType = convertToMrtTensorType(ctx, rankedType);
        if (newType != oldType) {
          arg.setType(newType);
        }
      }
    }

    // Update function signature
    auto funcType = funcOp.getFunctionType();
    SmallVector<Type> newInputTypes;
    SmallVector<Type> newResultTypes;

    for (auto inputType : funcType.getInputs()) {
      if (auto rankedType = dyn_cast<RankedTensorType>(inputType)) {
        newInputTypes.push_back(convertToMrtTensorType(ctx, rankedType));
      } else {
        newInputTypes.push_back(inputType);
      }
    }

    for (auto resultType : funcType.getResults()) {
      if (auto rankedType = dyn_cast<RankedTensorType>(resultType)) {
        newResultTypes.push_back(convertToMrtTensorType(ctx, rankedType));
      } else {
        newResultTypes.push_back(resultType);
      }
    }

    auto newFuncType = mlir::FunctionType::get(ctx, newInputTypes, newResultTypes);
    funcOp.setFunctionType(newFuncType);
  });
}

}  // namespace

namespace mlir {

#define GEN_PASS_DEF_CONVERTRANKEDTENSORTOMRTTENSOR
#include "mopt/Dialect/Mrt/Transforms/Passes.h.inc"

struct ConvertRankedTensorToMrtTensorPass
    : public impl::ConvertRankedTensorToMrtTensorBase<ConvertRankedTensorToMrtTensorPass> {
  using ConvertRankedTensorToMrtTensorBase::ConvertRankedTensorToMrtTensorBase;

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mrt::MrtDialect>();
    registry.insert<mlir::func::FuncDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    convertAllTypesToMrtTensorType(module);
  }
};

std::unique_ptr<Pass> createConvertRankedTensorToMrtTensorPass() {
  return std::make_unique<ConvertRankedTensorToMrtTensorPass>();
}

}  // namespace mlir
