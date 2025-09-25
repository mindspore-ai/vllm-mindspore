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

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/Builders.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"

namespace mrt {
namespace pass {
// Inherit from OperationPass<func::FuncOp>, with explicit namespace: mlir::func::FuncOp
// TODO(dayschan) remove this pass.
struct ReplaceAddWithMulPass
    : public mlir::PassWrapper<ReplaceAddWithMulPass, mlir::OperationPass<mlir::func::FuncOp>> {
  // cppcheck-suppress unknownMacro
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ReplaceAddWithMulPass)

  mlir::StringRef getArgument() const final { return "replace-tosa-add-with-mul"; }
  mlir::StringRef getDescription() const final { return "Replace all tosa.add with tosa.mul (demo only)."; }

  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();
    mlir::OpBuilder builder(func.getContext());
    llvm::SmallVector<mlir::Operation *, 8> toErase;

    func.walk([&](mlir::Operation *op) {
      if (auto add = llvm::dyn_cast<mlir::tosa::AddOp>(op)) {
        builder.setInsertionPoint(add);

        mlir::Value lhs = add.getInput1();
        mlir::Value rhs = add.getInput2();
        mlir::Type outTy = add.getType();

        // Construct an i32 scalar constant (ElementsAttr) with shift = 0, as the third input of tosa.mul
        mlir::Location loc = add.getLoc();
        // mlir::Type i32Ty = builder.getI32Type();
        // mlir::RankedTensorType shiftTy = mlir::RankedTensorType::get({}, i32Ty);

        // // DenseElementsAttr is a subclass of ElementsAttr, satisfying ConstOp's value parameter requirement
        // mlir::IntegerAttr zeroAttr = builder.getI32IntegerAttr(0);
        // mlir::DenseElementsAttr shiftDense = mlir::DenseElementsAttr::get(shiftTy, zeroAttr);
        // mlir::ElementsAttr shiftElems = shiftDense;

        // mlir::Value shiftVal = builder.create<mlir::tosa::ConstOp>(loc, shiftTy, shiftElems).getResult();

        // auto mul = builder.create<mlir::tosa::MulOp>(loc, outTy, lhs, rhs, shiftVal);
        auto mul = builder.create<mlir::tosa::MulOp>(loc, outTy, lhs, rhs, 0);

        add.getResult().replaceAllUsesWith(mul.getResult());
        toErase.push_back(add);
      }
    });

    for (mlir::Operation *op : toErase) {
      op->erase();
    }
  }
};

// Factory function for explicit external creation
std::unique_ptr<mlir::Pass> createReplaceAddWithMulPass() { return std::make_unique<ReplaceAddWithMulPass>(); }

}  // namespace pass
}  // namespace mrt

// Static registration (for invocation/loading via registration name)
static mlir::PassRegistration<mrt::pass::ReplaceAddWithMulPass> pass;
