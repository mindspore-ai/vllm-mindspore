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

#include <cstdint>
#include <string>
#include <vector>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"

#include "mopt/Dialect/Mrt/Mrt.h"
#include "mopt/Dialect/Mrt/MrtDialect.h"
#include "mopt/Dialect/Mrt/Transforms/Passes.h"

using mlir::MLIRContext;
using mlir::ModuleOp;
using mlir::OpBuilder;
using mlir::Operation;
using mlir::OperationPass;
using mlir::RankedTensorType;
using mlir::Type;
using mlir::Value;

namespace {

// Helper function to create device attribute with optional index
mrt::DeviceAttr createDeviceAttr(MLIRContext *ctx, const std::string &deviceType, int64_t index = -1) {
  auto typeAttr = mlir::StringAttr::get(ctx, deviceType);
  auto indexAttr = mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64), index);
  return mrt::DeviceAttr::get(ctx, typeAttr, indexAttr);
}

// Helper function to extract device from tensor type
mrt::DeviceAttr getDeviceFromType(Type type) {
  if (auto tensorType = mlir::dyn_cast<mrt::TensorType>(type)) {
    return tensorType.getDevice();
  }
  return nullptr;
}

// Get device from first input tensor that has device info
mrt::DeviceAttr getDeviceFromFirstInput(Operation *op) {
  for (auto operand : op->getOperands()) {
    auto device = getDeviceFromType(operand.getType());
    if (device) {
      return device;
    }
  }
  return nullptr;
}

}  // namespace

namespace mlir {

#define GEN_PASS_DEF_SETTENSORDEVICE
#include "mopt/Dialect/Mrt/Transforms/Passes.h.inc"

struct SetTensorDevicePass : public impl::SetTensorDeviceBase<SetTensorDevicePass> {
  using SetTensorDeviceBase::SetTensorDeviceBase;

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mrt::MrtDialect>();
    registry.insert<mlir::func::FuncDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = module.getContext();

    std::string deviceTypeStr = deviceType;
    int64_t deviceIdx = deviceIndex;

    module.walk([&](mlir::func::FuncOp funcOp) {
      llvm::DenseMap<Value, mrt::DeviceAttr> valueToDevice;
      llvm::SmallVector<std::pair<Operation *, mrt::DeviceAttr>> opsToUpdate;

      initializeFunctionArguments(funcOp, ctx, deviceTypeStr, deviceIdx, valueToDevice);
      collectOpsToUpdate(funcOp, ctx, valueToDevice, opsToUpdate);
      rewriteOperationResults(ctx, opsToUpdate);
      updateFunctionSignature(funcOp, ctx);
    });
  }

 private:
  static void initializeFunctionArguments(mlir::func::FuncOp funcOp, MLIRContext *ctx, const std::string &deviceTypeStr,
                                          int64_t deviceIdx, llvm::DenseMap<Value, mrt::DeviceAttr> &valueToDevice) {
    auto funcArgs = funcOp.getArguments();
    mrt::DeviceAttr defaultDevice = createDeviceAttr(ctx, deviceTypeStr, deviceIdx);

    for (auto arg : funcArgs) {
      Type argType = arg.getType();
      Type newType = nullptr;

      if (auto tensorType = mlir::dyn_cast<mrt::TensorType>(argType)) {
        if (!tensorType.getDevice()) {
          newType = mrt::TensorType::get(ctx, tensorType.getShape(), tensorType.getElementType(), defaultDevice);
        }
      } else if (auto rankedType = mlir::dyn_cast<RankedTensorType>(argType)) {
        newType = mrt::TensorType::get(ctx, rankedType.getShape(), rankedType.getElementType(), defaultDevice);
      }

      if (newType) {
        valueToDevice[arg] = defaultDevice;
        arg.setType(newType);
        continue;
      }

      auto device = getDeviceFromType(argType);
      if (device) {
        valueToDevice[arg] = device;
      } else {
        valueToDevice[arg] = defaultDevice;
      }
    }
  }

  static void collectOpsToUpdate(mlir::func::FuncOp funcOp, MLIRContext *ctx,
                                 llvm::DenseMap<Value, mrt::DeviceAttr> &valueToDevice,
                                 llvm::SmallVector<std::pair<Operation *, mrt::DeviceAttr>> &opsToUpdate) {
    funcOp.walk([&](Operation *op) {
      if (mlir::isa<mlir::func::FuncOp>(op)) {
        return;
      }

      mrt::DeviceAttr device;
      for (auto operand : op->getOperands()) {
        auto it = valueToDevice.find(operand);
        if (it != valueToDevice.end()) {
          device = it->second;
          break;
        }
        auto typeDevice = getDeviceFromType(operand.getType());
        if (typeDevice) {
          device = typeDevice;
          break;
        }
      }

      if (!device) {
        for (auto arg : funcOp.getArguments()) {
          auto argDevice = getDeviceFromType(arg.getType());
          if (argDevice) {
            device = argDevice;
            break;
          }
        }
      }

      if (!device) {
        device = createDeviceAttr(ctx, "cpu");
      }

      bool needsUpdate = false;
      for (auto result : op->getResults()) {
        valueToDevice[result] = device;

        Type type = result.getType();
        if (auto tensorType = mlir::dyn_cast<mrt::TensorType>(type)) {
          if (!tensorType.getDevice()) {
            needsUpdate = true;
          }
        } else if (mlir::isa<RankedTensorType>(type)) {
          needsUpdate = true;
        }
      }

      if (needsUpdate) {
        opsToUpdate.push_back({op, device});
      }
    });
  }

  static void rewriteOperationResults(MLIRContext *ctx,
                                      llvm::SmallVector<std::pair<Operation *, mrt::DeviceAttr>> &opsToUpdate) {
    OpBuilder builder(ctx);

    for (auto it = opsToUpdate.rbegin(); it != opsToUpdate.rend(); ++it) {
      Operation *op = it->first;
      mrt::DeviceAttr device = it->second;

      if (op->getNumResults() == 0) {
        continue;
      }

      llvm::SmallVector<Type> newResultTypes;
      bool hasTypeChange = false;
      for (auto result : op->getResults()) {
        Type oldType = result.getType();
        Type newType = oldType;

        if (auto tensorType = mlir::dyn_cast<mrt::TensorType>(oldType)) {
          if (!tensorType.getDevice()) {
            newType = mrt::TensorType::get(ctx, tensorType.getShape(), tensorType.getElementType(), device);
            hasTypeChange = true;
          }
        } else if (auto rankedType = mlir::dyn_cast<RankedTensorType>(oldType)) {
          newType = mrt::TensorType::get(ctx, rankedType.getShape(), rankedType.getElementType(), device);
          hasTypeChange = true;
        }

        newResultTypes.push_back(newType);
      }

      if (!hasTypeChange) {
        continue;
      }

      mlir::OperationState state(op->getLoc(), op->getName());
      state.addOperands(op->getOperands());
      state.addAttributes(op->getAttrs());
      state.addTypes(newResultTypes);

      Operation *newOp = Operation::create(state);
      op->getBlock()->getOperations().insert(op->getIterator(), newOp);

      for (auto [oldResult, newResult] : llvm::zip(op->getResults(), newOp->getResults())) {
        oldResult.replaceAllUsesWith(newResult);
      }

      op->erase();
    }
  }

  static void updateFunctionSignature(mlir::func::FuncOp funcOp, MLIRContext *ctx) {
    llvm::SmallVector<Type> newInputTypes;
    llvm::transform(funcOp.getArguments(), std::back_inserter(newInputTypes),
                    [](mlir::BlockArgument arg) { return arg.getType(); });

    llvm::SmallVector<Type> newResultTypes;
    auto returnOp = funcOp.getBody().front().getTerminator();
    if (auto funcReturnOp = mlir::dyn_cast<mlir::func::ReturnOp>(returnOp)) {
      llvm::transform(funcReturnOp.getOperands(), std::back_inserter(newResultTypes),
                      [](mlir::Value operand) { return operand.getType(); });
    } else {
      auto originalType = funcOp.getFunctionType();
      llvm::copy(originalType.getResults(), std::back_inserter(newResultTypes));
    }

    auto newFunctionType = mlir::FunctionType::get(ctx, newInputTypes, newResultTypes);
    funcOp.setFunctionType(newFunctionType);
  }
};

std::unique_ptr<Pass> createSetTensorDevice() { return std::make_unique<SetTensorDevicePass>(); }

}  // namespace mlir
