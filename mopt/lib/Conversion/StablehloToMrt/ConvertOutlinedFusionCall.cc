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

#include "mopt/Conversion/StablehloToMrt/StablehloToMrt.h"
#include "mopt/Conversion/StablehloToMrt/StablehloToMrtTypeConverter.h"
#include "mopt/Conversion/StablehloToMrt/OutlinedFusionCall.h"
#include "mopt/Dialect/Mrt/Mrt.h"
#include "mopt/Dialect/Mrt/MrtDialect.h"

#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "stablehlo/conversions/linalg/transforms/Passes.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "convert-outlined-fusion-call"

using mlir::cast;
using mlir::dyn_cast;
using mlir::failed;
using mlir::func::FuncOp;
using mlir::MLIRContext;
using mlir::ModuleOp;
using mlir::OpBuilder;
using mlir::OperationPass;
using mlir::Pass;
using mlir::PassWrapper;
using mlir::SmallVector;
using mlir::StringRef;

namespace {

//===----------------------------------------------------------------------===//
// Serialization utilities
//===----------------------------------------------------------------------===//

/// Serialize a StableHLO function to MLIR text format with Linalg conversion and hacc attributes.
std::string serializeFunc(FuncOp func) {
  MLIRContext *ctx = func.getContext();

  // Create a standalone module for serialization
  mlir::OwningOpRef<ModuleOp> tempModule = ModuleOp::create(func.getLoc());
  OpBuilder moduleBuilder(tempModule->getBodyRegion());

  // Clone function
  auto clonedFunc = cast<FuncOp>(moduleBuilder.clone(*func.getOperation()));

  // Remove fusion.outlined attribute and add hacc attributes
  clonedFunc->removeAttr("fusion.outlined");
  clonedFunc->setAttr("hacc.entry", mlir::UnitAttr::get(ctx));

  // Add hacc.function_kind<HOST> attribute
  mlir::Attribute functionKindAttr = mlir::parseAttribute("#hacc.function_kind<HOST>", ctx);
  if (functionKindAttr) {
    clonedFunc->setAttr("hacc.function_kind", functionKindAttr);
  }

  clonedFunc.setPublic();

  // Run stablehlo-legalize-to-linalg on the temp module
  mlir::PassManager pm(ctx);
  pm.addPass(mlir::stablehlo::createStablehloLegalizeToLinalgPass());
  if (failed(pm.run(tempModule.get()))) {
    return "";
  }

  // Serialize to string
  std::string mlirText;
  llvm::raw_string_ostream os(mlirText);
  tempModule->print(os);

  return mlirText;
}

using SerializedFuncMap = llvm::DenseMap<mlir::StringAttr, std::string>;

//===----------------------------------------------------------------------===//
// Conversion pattern: func.call -> mrt.linalg_call
//===----------------------------------------------------------------------===//

/// Convert calls to outlined fusion functions into mrt.linalg_call.
struct ConvertOutlinedFusionCallOp : public mlir::OpConversionPattern<mlir::func::CallOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(mlir::func::CallOp callOp, OpAdaptor adaptor,
                                      mlir::ConversionPatternRewriter &rewriter) const override {
    auto mlirTextAttr =
        callOp->getAttrOfType<mlir::StringAttr>(mopt::kOutlinedFusionMlirTextAttr);
    if (!mlirTextAttr) {
      return rewriter.notifyMatchFailure(callOp, "missing outlined fusion MLIR text attribute");
    }

    // Create mlir_text operand as mrt.string
    auto stringType = mrt::StringType::get(rewriter.getContext());
    mlir::Value mlirTextValue =
        rewriter.create<mrt::CreateStringOp>(callOp.getLoc(), stringType, mlirTextAttr);

    // Convert result types using TypeConverter: tensor<...> -> !mrt.tensor<...>
    llvm::SmallVector<mlir::Type> convertedResultTypes;
    if (failed(getTypeConverter()->convertTypes(callOp.getResultTypes(), convertedResultTypes))) {
      return rewriter.notifyMatchFailure(callOp, "failed to convert result types");
    }

    // Convert operands using TypeConverter with target materialization
    llvm::SmallVector<mlir::Value> convertedOperands;
    convertedOperands.reserve(adaptor.getOperands().size());
    for (mlir::Value operand : adaptor.getOperands()) {
      mlir::Type srcType = operand.getType();
      mlir::Type dstType = getTypeConverter()->convertType(srcType);
      if (!dstType) {
        return rewriter.notifyMatchFailure(callOp, "failed to convert operand type");
      }

      if (srcType != dstType) {
        // Use TypeConverter's target materialization to insert cast
        mlir::Value materialized = getTypeConverter()->materializeTargetConversion(
            rewriter, callOp.getLoc(), dstType, operand);
        if (!materialized) {
          return rewriter.notifyMatchFailure(callOp, "failed to materialize operand to MRT type");
        }
        convertedOperands.push_back(materialized);
      } else {
        convertedOperands.push_back(operand);
      }
    }

    // Create mrt.linalg_call with MRT tensor types
    auto linalgCallOp = rewriter.create<mrt::LinalgCallOp>(
        callOp.getLoc(), convertedResultTypes, mlirTextValue, convertedOperands);

    rewriter.replaceOp(callOp, linalgCallOp.getResults());
    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertOutlinedFusionCallPass
//===----------------------------------------------------------------------===//

struct ConvertOutlinedFusionCallPass
    : public PassWrapper<ConvertOutlinedFusionCallPass, OperationPass<ModuleOp>> {
  // cppcheck-suppress unknownMacro
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertOutlinedFusionCallPass)

  StringRef getArgument() const final { return "convert-outlined-fusion-call"; }

  StringRef getDescription() const final {
    return "Serialize outlined fusion functions to Linalg MLIR text and convert calls to mrt.linalg_call";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::func::FuncDialect>();
    registry.insert<mlir::stablehlo::StablehloDialect>();
    registry.insert<mlir::linalg::LinalgDialect>();
    registry.insert<mlir::arith::ArithDialect>();
    registry.insert<mlir::math::MathDialect>();
    registry.insert<mlir::tensor::TensorDialect>();
    registry.insert<mrt::MrtDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();

    LLVM_DEBUG(llvm::dbgs() << "[ConvertOutlinedFusionCall] Starting pass\n");

    // ===== Phase 1: Serialize outlined fusion functions and annotate calls =====
    SmallVector<FuncOp> outlinedFuncs;
    SerializedFuncMap serializedFuncs;

    module.walk([&](FuncOp funcOp) {
      if (funcOp->hasAttr("fusion.outlined")) {
        outlinedFuncs.push_back(funcOp);
      }
    });

    if (outlinedFuncs.empty()) {
      return;
    }

    LLVM_DEBUG(llvm::dbgs() << "[ConvertOutlinedFusionCall] Found " << outlinedFuncs.size()
                            << " outlined functions to convert\n");

    // Serialize all outlined functions
    for (FuncOp funcOp : outlinedFuncs) {
      std::string mlirText = serializeFunc(funcOp);
      if (mlirText.empty()) {
        LLVM_DEBUG(llvm::dbgs() << "[ConvertOutlinedFusionCall] Failed to serialize outlined function: "
                                << funcOp.getName() << "\n");
        signalPassFailure();
        return;
      }
      serializedFuncs[funcOp.getNameAttr()] = std::move(mlirText);
    }

    // Annotate all call sites with serialized MLIR text
    int annotatedCalls = 0;
    module.walk([&](mlir::func::CallOp callOp) {
      auto callee = dyn_cast<mlir::SymbolRefAttr>(callOp.getCallableForCallee());
      if (!callee)
        return;
      auto it = serializedFuncs.find(callee.getRootReference());
      if (it == serializedFuncs.end())
        return;
      callOp->setAttr(mopt::kOutlinedFusionMlirTextAttr, mlir::StringAttr::get(ctx, it->second));
      annotatedCalls++;
    });

    LLVM_DEBUG(llvm::dbgs() << "[ConvertOutlinedFusionCall] Annotated " << annotatedCalls
                            << " calls with " << mopt::kOutlinedFusionMlirTextAttr << "\n");

    // ===== Phase 2: Convert annotated func.call to mrt.linalg_call =====
    mopt::StablehloToMrtTypeConverter typeConverter(ctx);

    mlir::ConversionTarget target(*ctx);
    target.addLegalDialect<mrt::MrtDialect>();
    target.addLegalDialect<mlir::arith::ArithDialect>();
    target.addLegalDialect<mlir::stablehlo::StablehloDialect>();
    target.addLegalOp<mlir::ModuleOp>();
    target.addLegalOp<mlir::func::FuncOp>();
    target.addLegalOp<mlir::func::ReturnOp>();
    target.addLegalOp<mlir::UnrealizedConversionCastOp>();

    // Annotated calls must be converted to mrt.linalg_call
    target.addDynamicallyLegalOp<mlir::func::CallOp>([](mlir::func::CallOp callOp) {
      return !callOp->hasAttr(mopt::kOutlinedFusionMlirTextAttr);
    });

    mlir::RewritePatternSet patterns(ctx);
    patterns.add<ConvertOutlinedFusionCallOp>(typeConverter, ctx);

    if (failed(mlir::applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }
};

}  // namespace

namespace mlir {

std::unique_ptr<Pass> createConvertOutlinedFusionCallPass() {
  return std::make_unique<ConvertOutlinedFusionCallPass>();
}

}  // namespace mlir
