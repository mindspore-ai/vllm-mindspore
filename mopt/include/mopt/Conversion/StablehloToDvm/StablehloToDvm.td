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

include "mlir/IR/PatternBase.td"
include "stablehlo/dialect/StablehloOps.td"
include "mopt/Dialect/Dvm/DvmOps.td"

//===----------------------------------------------------------------------===//
// Unary Ops
//===----------------------------------------------------------------------===//

// Define a pattern to convert StableHLO Unary Ops to Dvm Unary Ops.
class StablehloToDvmUnaryOp<Op stablehloOp, I32EnumAttrCase dvmOpType> : Pat<
  (stablehloOp $input),
  (Dvm_UnaryOp dvmOpType, $input)
>;

// Define a pattern for StableHLO Ops with ResultAccuracyAttr (e.g. Sqrt, Exp, Log)
// These ops have two arguments: operand and result_accuracy
class StablehloToDvmUnaryOpWithAccuracy<Op stablehloOp, I32EnumAttrCase dvmOpType> : Pat<
  (stablehloOp $input, $accuracy),
  (Dvm_UnaryOp dvmOpType, $input)
>;

// Sqrt
def : StablehloToDvmUnaryOpWithAccuracy<StableHLO_SqrtOp, Dvm_UnaryOpSqrt>;

// Abs
def : StablehloToDvmUnaryOp<StableHLO_AbsOp, Dvm_UnaryOpAbs>;

// Exp
def : StablehloToDvmUnaryOpWithAccuracy<StableHLO_ExpOp, Dvm_UnaryOpExp>;

// Log
def : StablehloToDvmUnaryOpWithAccuracy<StableHLO_LogOp, Dvm_UnaryOpLog>;

// Reciprocal
// Note: StableHLO does not have a direct ReciprocalOp. It is usually represented as DivOp(1, x).
// We do not convert DivOp(1, x) to Reciprocal here yet.
// Refer to: mopt/lib/Conversion/StablehloToDvm/StablehloToDvm.cc

// IsFinite
def : StablehloToDvmUnaryOp<StableHLO_IsFiniteOp, Dvm_UnaryOpIsFinite>;

// LogicalNot
def : StablehloToDvmUnaryOp<StableHLO_NotOp, Dvm_UnaryOpLogicalNot>;

// Round
// Mapping RoundNearestEvenOp to Round.
def : StablehloToDvmUnaryOp<StableHLO_RoundNearestEvenOp, Dvm_UnaryOpRound>;

// Floor
def : StablehloToDvmUnaryOp<StableHLO_FloorOp, Dvm_UnaryOpFloor>;

// Ceil
def : StablehloToDvmUnaryOp<StableHLO_CeilOp, Dvm_UnaryOpCeil>;

// Trunc
// Note: StableHLO does not have a direct TruncOp.

//===----------------------------------------------------------------------===//
// Binary Ops
//===----------------------------------------------------------------------===//

// Define a pattern to convert StableHLO Binary Ops to Dvm Binary Ops.
class StablehloToDvmBinaryOp<Op stablehloOp, I32EnumAttrCase dvmOpType> : Pat<
  (stablehloOp $lhs, $rhs),
  (Dvm_BinaryOp dvmOpType, $lhs, $rhs)
>;

// Add
def : StablehloToDvmBinaryOp<StableHLO_AddOp, Dvm_BinaryOpAdd>;

// Sub
def : StablehloToDvmBinaryOp<StableHLO_SubtractOp, Dvm_BinaryOpSub>;

// Mul
def : StablehloToDvmBinaryOp<StableHLO_MulOp, Dvm_BinaryOpMul>;

// Div
def : StablehloToDvmBinaryOp<StableHLO_DivOp, Dvm_BinaryOpDiv>;

// Pow
def : StablehloToDvmBinaryOp<StableHLO_PowOp, Dvm_BinaryOpPow>;

// Maximum
def : StablehloToDvmBinaryOp<StableHLO_MaxOp, Dvm_BinaryOpMaximum>;

// Minimum
def : StablehloToDvmBinaryOp<StableHLO_MinOp, Dvm_BinaryOpMinimum>;

// LogicalAnd
def : StablehloToDvmBinaryOp<StableHLO_AndOp, Dvm_BinaryOpLogicalAnd>;

// LogicalOr
def : StablehloToDvmBinaryOp<StableHLO_OrOp, Dvm_BinaryOpLogicalOr>;
