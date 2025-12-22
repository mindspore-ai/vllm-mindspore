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

#include "mopt/Conversion/StablehloToDvm/StablehloToDvm.h"

#include "stablehlo/dialect/StablehloOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "mopt/Dialect/Dvm/Dvm.h"
#include "mopt/Dialect/Dvm/DvmDialect.h"

namespace mlir {
namespace {

// Include the auto-generated patterns
#include "mopt/Conversion/StablehloToDvm/StablehloToDvm.inc"

struct ConvertReciprocalOp : public OpConversionPattern<mlir::stablehlo::DivOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::DivOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();

    // Check if shapes match
    auto lhsType = dyn_cast<ShapedType>(lhs.getType());
    auto rhsType = dyn_cast<ShapedType>(rhs.getType());
    if (!lhsType || !rhsType || lhsType != rhsType) return failure();

    // Check if lhs comes from a ConstantOp
    auto constOp = lhs.getDefiningOp<mlir::stablehlo::ConstantOp>();
    if (!constOp) return failure();

    auto constVal = dyn_cast<DenseElementsAttr>(constOp.getValue());
    if (!constVal) return failure();
    if (!constVal.isSplat()) return failure();

    if (auto floatType = dyn_cast<FloatType>(constVal.getElementType())) {
      if (!constVal.getSplatValue<APFloat>().isExactlyValue(1.0)) return failure();
    } else if (auto intType = dyn_cast<IntegerType>(constVal.getElementType())) {
      if (constVal.getSplatValue<APInt>() != 1) return failure();
    } else {
      return failure();
    }

    rewriter.replaceOpWithNewOp<dvm::UnaryOp>(
      op, op.getResult().getType(), dvm::UnaryOpTypeAttr::get(getContext(), dvm::UnaryOpType::Reciprocal), rhs);

    // If constant op has no other users, erase it.
    // Note: Since we just replaced 'op' (which used constOp), the use count of constOp should have decreased.
    // However, PatternRewriter's replacement logic might not immediately update use lists in a way that use_empty()
    // reflects the removal of the current op's usage until the end of the pattern application or transaction.
    // But standard rewriter.eraseOp/replaceOp usually updates the IR.
    // A common pitfall is that 'op' is still alive (just pending erasure) and still using 'constOp'.
    // To safely check if we can remove 'constOp', we need to see if 'op' was its *only* user.
    if (constOp->hasOneUse()) {
      rewriter.eraseOp(constOp);
    }

    return success();
  }
};

struct ConvertCompareOp : public OpConversionPattern<mlir::stablehlo::CompareOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::CompareOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();

    using mlir::stablehlo::ComparisonDirection;
    static const std::pair<ComparisonDirection, dvm::BinaryOpType> map[] = {
      {ComparisonDirection::EQ, dvm::BinaryOpType::Equal},   {ComparisonDirection::NE, dvm::BinaryOpType::NotEqual},
      {ComparisonDirection::GT, dvm::BinaryOpType::Greater}, {ComparisonDirection::GE, dvm::BinaryOpType::GreaterEqual},
      {ComparisonDirection::LT, dvm::BinaryOpType::Less},    {ComparisonDirection::LE, dvm::BinaryOpType::LessEqual},
    };

    ComparisonDirection direction = op.getComparisonDirection();
    for (const auto &entry : map) {
      if (entry.first == direction) {
        dvm::BinaryOpType dvmOpType = entry.second;
        rewriter.replaceOpWithNewOp<dvm::BinaryOp>(op, op.getResult().getType(),
                                                   dvm::BinaryOpTypeAttr::get(getContext(), dvmOpType), lhs, rhs);
        return success();
      }
    }

    return failure();
  }
};

struct ConvertDotOp : public OpConversionPattern<mlir::stablehlo::DotOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::DotOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();
    auto lhsType = dyn_cast<RankedTensorType>(lhs.getType());
    auto rhsType = dyn_cast<RankedTensorType>(rhs.getType());
    auto resultType = dyn_cast<RankedTensorType>(op.getType());

    if (!lhsType || !rhsType || !resultType) {
      return failure();
    }

    int64_t lhsRank = lhsType.getRank();
    int64_t rhsRank = rhsType.getRank();

    Value newLhs = lhs;
    Value newRhs = rhs;
    bool reshapeResult = false;

    // StableHLO DotOp logic:
    // (1, 1) -> scalar
    // (2, 1) -> vector
    // (1, 2) -> vector
    // (2, 2) -> matrix

    if (lhsRank == 2 && rhsRank == 2) {
      // Direct pass-through
    } else if (lhsRank == 1 && rhsRank == 1) {
      // Vector-Vector
      SmallVector<int64_t> lhsShape = {1, lhsType.getDimSize(0)};
      SmallVector<int64_t> rhsShape = {rhsType.getDimSize(0), 1};

      newLhs = rewriter.create<dvm::ReshapeOp>(op.getLoc(), RankedTensorType::get(lhsShape, lhsType.getElementType()),
                                               lhs, rewriter.getI64ArrayAttr(lhsShape));
      newRhs = rewriter.create<dvm::ReshapeOp>(op.getLoc(), RankedTensorType::get(rhsShape, rhsType.getElementType()),
                                               rhs, rewriter.getI64ArrayAttr(rhsShape));
      reshapeResult = true;
    } else if (lhsRank == 2 && rhsRank == 1) {
      // Matrix-Vector
      SmallVector<int64_t> rhsShape = {rhsType.getDimSize(0), 1};
      newRhs = rewriter.create<dvm::ReshapeOp>(op.getLoc(), RankedTensorType::get(rhsShape, rhsType.getElementType()),
                                               rhs, rewriter.getI64ArrayAttr(rhsShape));
      reshapeResult = true;
    } else if (lhsRank == 1 && rhsRank == 2) {
      // Vector-Matrix
      SmallVector<int64_t> lhsShape = {1, lhsType.getDimSize(0)};
      newLhs = rewriter.create<dvm::ReshapeOp>(op.getLoc(), RankedTensorType::get(lhsShape, lhsType.getElementType()),
                                               lhs, rewriter.getI64ArrayAttr(lhsShape));
      reshapeResult = true;
    } else {
      return failure();
    }

    auto newLhsType = cast<RankedTensorType>(newLhs.getType());
    auto newRhsType = cast<RankedTensorType>(newRhs.getType());

    int64_t M = newLhsType.getDimSize(newLhsType.getRank() - 2);
    int64_t N = newRhsType.getDimSize(newRhsType.getRank() - 1);
    SmallVector<int64_t> matMulResultShape = {M, N};

    auto matMulResultType = RankedTensorType::get(matMulResultShape, resultType.getElementType());

    Value matmul = rewriter.create<dvm::MatMulOp>(op.getLoc(), matMulResultType, newLhs, newRhs,
                                                  rewriter.getBoolAttr(false), rewriter.getBoolAttr(false), nullptr);

    if (reshapeResult) {
      rewriter.replaceOpWithNewOp<dvm::ReshapeOp>(op, resultType, matmul,
                                                  rewriter.getI64ArrayAttr(resultType.getShape()));
    } else {
      rewriter.replaceOp(op, matmul);
    }
    return success();
  }
};

struct ConvertDotGeneralOp : public OpConversionPattern<mlir::stablehlo::DotGeneralOp> {
  using OpConversionPattern::OpConversionPattern;

  struct DotGeneralInfo {
    bool transA;
    bool transB;
    int64_t M, N, K_lhs, K_rhs;
    SmallVector<int64_t> lhsFlatShape;
    SmallVector<int64_t> rhsFlatShape;
    SmallVector<int64_t> resultShape;
  };

  LogicalResult matchAndRewrite(mlir::stablehlo::DotGeneralOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();
    auto lhsType = dyn_cast<RankedTensorType>(lhs.getType());
    auto rhsType = dyn_cast<RankedTensorType>(rhs.getType());
    auto resultType = dyn_cast<RankedTensorType>(op.getType());

    if (!lhsType || !rhsType || !resultType) {
      return failure();
    }

    DotGeneralInfo info;
    if (failed(analyze(op, lhsType, rhsType, resultType, info))) {
      return failure();
    }

    // Reshape Inputs
    Value newLhs = lhs;
    Value newRhs = rhs;

    if (lhsType.getShape() != ArrayRef<int64_t>(info.lhsFlatShape)) {
      newLhs =
        rewriter.create<dvm::ReshapeOp>(op.getLoc(), RankedTensorType::get(info.lhsFlatShape, lhsType.getElementType()),
                                        lhs, rewriter.getI64ArrayAttr(info.lhsFlatShape));
    }
    if (rhsType.getShape() != ArrayRef<int64_t>(info.rhsFlatShape)) {
      newRhs =
        rewriter.create<dvm::ReshapeOp>(op.getLoc(), RankedTensorType::get(info.rhsFlatShape, rhsType.getElementType()),
                                        rhs, rewriter.getI64ArrayAttr(info.rhsFlatShape));
    }

    auto matMulResultType = RankedTensorType::get(info.resultShape, resultType.getElementType());

    Value matmul =
      rewriter.create<dvm::MatMulOp>(op.getLoc(), matMulResultType, newLhs, newRhs, rewriter.getBoolAttr(info.transA),
                                     rewriter.getBoolAttr(info.transB), nullptr);

    if (resultType.getShape() != ArrayRef<int64_t>(info.resultShape)) {
      rewriter.replaceOpWithNewOp<dvm::ReshapeOp>(op, resultType, matmul,
                                                  rewriter.getI64ArrayAttr(resultType.getShape()));
    } else {
      rewriter.replaceOp(op, matmul);
    }

    return success();
  }

  LogicalResult analyze(mlir::stablehlo::DotGeneralOp op, RankedTensorType lhsType, RankedTensorType rhsType,
                        RankedTensorType resType, DotGeneralInfo &info) const {
    auto dotNums = op.getDotDimensionNumbers();
    auto lhsBatchDims = dotNums.getLhsBatchingDimensions();
    auto rhsBatchDims = dotNums.getRhsBatchingDimensions();

    // Inlined validation logic
    for (size_t i = 0; i < lhsBatchDims.size(); ++i) {
      if (lhsBatchDims[i] != (int64_t)i || rhsBatchDims[i] != (int64_t)i) {
        return failure();
      }
    }
    if (lhsBatchDims.size() != rhsBatchDims.size()) {
      return failure();
    }

    int64_t batchRank = lhsBatchDims.size();

    // Analyze LHS
    if (failed(analyzeOperand(lhsType, batchRank, dotNums.getLhsContractingDimensions(), info.transA, info.M,
                              info.K_lhs, info.lhsFlatShape))) {
      return failure();
    }

    // Analyze RHS
    bool rhsHelperTrans;
    int64_t N, K_rhs;
    if (failed(analyzeOperand(rhsType, batchRank, dotNums.getRhsContractingDimensions(), rhsHelperTrans, N, K_rhs,
                              info.rhsFlatShape))) {
      return failure();
    }

    info.transB = !rhsHelperTrans;  // Invert logic for RHS
    info.N = N;
    info.K_rhs = K_rhs;

    if (info.K_lhs != info.K_rhs) {
      return failure();
    }

    for (int64_t i = 0; i < batchRank; ++i) {
      info.resultShape.push_back(resType.getDimSize(i));
    }
    info.resultShape.push_back(info.M);
    info.resultShape.push_back(info.N);

    return success();
  }

  LogicalResult analyzeOperand(RankedTensorType type, int64_t batchRank, ArrayRef<int64_t> contractingDims, bool &trans,
                               int64_t &dim1, int64_t &dim2, SmallVector<int64_t> &flatShape) const {
    int64_t rank = type.getRank();
    SmallVector<int64_t> nonContractingIndices;
    for (int64_t i = batchRank; i < rank; ++i) {
      if (std::find(contractingDims.begin(), contractingDims.end(), i) == contractingDims.end()) {
        nonContractingIndices.push_back(i);
      }
    }

    bool isStandard = true;
    int64_t expected = batchRank;
    for (auto i : nonContractingIndices) {
      if (i != expected++) {
        isStandard = false;
      }
    }
    for (auto i : contractingDims) {
      if (i != expected++) {
        isStandard = false;
      }
    }

    bool isTransposed = true;
    expected = batchRank;
    for (auto i : contractingDims) {
      if (i != expected++) {
        isTransposed = false;
      }
    }
    for (auto i : nonContractingIndices) {
      if (i != expected++) {
        isTransposed = false;
      }
    }

    if (isStandard) {
      trans = false;
    } else if (isTransposed) {
      trans = true;
    } else {
      return failure();
    }

    dim1 = 1;
    for (auto i : nonContractingIndices) {
      dim1 *= type.getDimSize(i);
    }
    dim2 = 1;
    for (auto i : contractingDims) {
      dim2 *= type.getDimSize(i);
    }

    for (int64_t i = 0; i < batchRank; ++i) {
      flatShape.push_back(type.getDimSize(i));
    }
    flatShape.push_back(trans ? dim2 : dim1);
    flatShape.push_back(trans ? dim1 : dim2);

    return success();
  }
};

struct ConvertStablehloToDvmPass : public PassWrapper<ConvertStablehloToDvmPass, OperationPass<ModuleOp>> {
  // cppcheck-suppress unknownMacro
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertStablehloToDvmPass)

  StringRef getArgument() const final { return "convert-stablehlo-to-dvm"; }
  StringRef getDescription() const final { return "Convert StableHLO operations to DVM dialect operations"; }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::stablehlo::StablehloDialect>();
    registry.insert<mlir::dvm::DvmDialect>();
    registry.insert<func::FuncDialect>();
  }

  static constexpr StringRef kFusionOutlinedAttr = "fusion.outlined";

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();

    // 0. Lift constants to arguments
    liftConstantsToArguments(module);

    // 1. Pre-process functions to insert dvm.load and dvm.store
    insertLoadStoreOps(module);

    ConversionTarget target(*ctx);
    target.addLegalDialect<mlir::dvm::DvmDialect>();
    target.addLegalDialect<func::FuncDialect>();

    // If op is inside an outlined-fusion-function, convert stablehlo ops to dvm ops.
    auto isInsideOutlinedFusion = [](Operation *op) {
      auto func = op->getParentOfType<func::FuncOp>();
      return func && func->hasAttr(kFusionOutlinedAttr);
    };

    target.addDynamicallyLegalDialect<mlir::stablehlo::StablehloDialect>(
      [&](Operation *op) { return !isInsideOutlinedFusion(op); });

    RewritePatternSet patterns(ctx);
    populateWithGenerated(patterns);
    patterns.add<ConvertCompareOp, ConvertDotOp, ConvertDotGeneralOp>(ctx);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }

  // Lift constants to arguments
  //
  // Example transformation:
  // Before:
  //   func @f(%arg0: tensor<4xf32>) {
  //     %c = stablehlo.constant dense<1.0> : tensor<4xf32>
  //     %res = stablehlo.add %arg0, %c : tensor<4xf32>
  //   }
  //   func @main(%arg0: tensor<4xf32>) {
  //     call @f(%arg0)
  //   }
  //
  // After:
  //   func @f(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) {
  //     %res = stablehlo.add %arg0, %arg1 : tensor<4xf32>
  //   }
  //   func @main(%arg0: tensor<4xf32>) {
  //     %c = stablehlo.constant dense<1.0> : tensor<4xf32>
  //     call @f(%arg0, %c)
  //   }
  void liftConstantsToArguments(ModuleOp module) {
    SymbolTable symbolTable(module);

    // 1. Collect functions to modify and their constants
    // Map: FuncOp -> List of ConstantOps to lift
    DenseMap<func::FuncOp, SmallVector<stablehlo::ConstantOp>> funcConstants;

    for (auto func : module.getOps<func::FuncOp>()) {
      if (func.isExternal()) continue;
      if (!func->hasAttr(kFusionOutlinedAttr)) continue;

      SmallVector<stablehlo::ConstantOp> constants;
      func.walk([&](stablehlo::ConstantOp constOp) { constants.push_back(constOp); });

      if (!constants.empty()) {
        funcConstants[func] = constants;
      }
    }

    // 2. Process each function
    for (auto [func, constants] : funcConstants) {
      // 2a. Update function signature
      SmallVector<Type> newArgTypes;
      for (auto constOp : constants) {
        newArgTypes.push_back(constOp.getType());
      }

      auto funcType = func.getFunctionType();
      SmallVector<Type> argTypes(funcType.getInputs());
      argTypes.append(newArgTypes.begin(), newArgTypes.end());

      func.setType(FunctionType::get(func.getContext(), argTypes, funcType.getResults()));

      // 2b. Add arguments to entry block
      Block &entryBlock = func.front();
      unsigned originalArgCount = entryBlock.getNumArguments();
      for (auto type : newArgTypes) {
        entryBlock.addArgument(type, func.getLoc());
      }

      // 2c. Update call sites (Must be done BEFORE erasing constants)
      auto uses = symbolTable.getSymbolUses(func, module);
      if (uses) {
        for (auto use : *uses) {
          if (auto callOp = dyn_cast<func::CallOp>(use.getUser())) {
            OpBuilder builder(callOp);
            SmallVector<Value> newOperands(callOp.getOperands());

            for (auto constOp : constants) {
              // Clone the constant at the call site
              auto newConst = builder.clone(*constOp.getOperation());
              newOperands.push_back(newConst->getResult(0));
            }

            callOp->setOperands(newOperands);
          }
        }
      }

      // 2d. Replace constant usages with new arguments and erase constants
      for (size_t i = 0; i < constants.size(); ++i) {
        auto constOp = constants[i];
        BlockArgument newArg = entryBlock.getArgument(originalArgCount + i);
        constOp.replaceAllUsesWith(newArg);
        constOp.erase();
      }
    }
  }

  // Pre-process functions to insert dvm.load and dvm.store
  //
  // Example transformation:
  // Before:
  //   func @f(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  //     %res = stablehlo.abs %arg0 : tensor<4xf32>
  //     return %res : tensor<4xf32>
  //   }
  //
  // After:
  //   func @f(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  //     %0 = dvm.load %arg0 : tensor<4xf32>
  //     %1 = stablehlo.abs %0 : tensor<4xf32>
  //     %2 = dvm.store %1 : tensor<4xf32>
  //     return %2 : tensor<4xf32>
  //   }
  void insertLoadStoreOps(ModuleOp module) {
    for (auto func : module.getOps<func::FuncOp>()) {
      if (func.isExternal()) continue;
      if (!func->hasAttr(kFusionOutlinedAttr)) continue;

      // Insert dvm.load for arguments
      if (!func.getBlocks().empty()) {
        Block &entryBlock = func.front();
        OpBuilder builder(&entryBlock, entryBlock.begin());

        for (auto arg : entryBlock.getArguments()) {
          if (isa<RankedTensorType>(arg.getType())) {
            auto loadOp = builder.create<dvm::LoadOp>(func.getLoc(), arg.getType(), arg);
            arg.replaceAllUsesExcept(loadOp.getResult(), loadOp);
          }
        }
      }

      // Insert dvm.store for return values
      func.walk([&](func::ReturnOp returnOp) {
        OpBuilder builder(returnOp);
        for (unsigned i = 0; i < returnOp.getNumOperands(); ++i) {
          Value operand = returnOp.getOperand(i);
          if (isa<RankedTensorType>(operand.getType())) {
            auto storeOp = builder.create<dvm::StoreOp>(returnOp.getLoc(), operand.getType(), operand);
            returnOp.setOperand(i, storeOp.getResult());
          }
        }
      });
    }
  }
};

}  // namespace

std::unique_ptr<Pass> createConvertStablehloToDvmPass() { return std::make_unique<ConvertStablehloToDvmPass>(); }

}  // namespace mlir
