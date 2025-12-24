// RUN: mopt-opt %s --convert-stablehlo-to-dvm | FileCheck %s

func.func @test_sqrt(%arg0: tensor<4xf32>) -> tensor<4xf32> attributes {fusion.outlined} {
  %0 = stablehlo.sqrt %arg0 : tensor<4xf32>
  return %0 : tensor<4xf32>
}
// CHECK-LABEL: func.func @test_sqrt
// CHECK: dvm.load
// CHECK: dvm.unary Sqrt
// CHECK: dvm.store

func.func @test_abs(%arg0: tensor<4xf32>) -> tensor<4xf32> attributes {fusion.outlined} {
  %0 = stablehlo.abs %arg0 : tensor<4xf32>
  return %0 : tensor<4xf32>
}
// CHECK-LABEL: func.func @test_abs
// CHECK: dvm.load
// CHECK: dvm.unary Abs
// CHECK: dvm.store

func.func @test_exp(%arg0: tensor<4xf32>) -> tensor<4xf32> attributes {fusion.outlined} {
  %0 = stablehlo.exponential %arg0 : tensor<4xf32>
  return %0 : tensor<4xf32>
}
// CHECK-LABEL: func.func @test_exp
// CHECK: dvm.load
// CHECK: dvm.unary Exp
// CHECK: dvm.store

func.func @test_log(%arg0: tensor<4xf32>) -> tensor<4xf32> attributes {fusion.outlined} {
  %0 = stablehlo.log %arg0 : tensor<4xf32>
  return %0 : tensor<4xf32>
}
// CHECK-LABEL: func.func @test_log
// CHECK: dvm.load
// CHECK: dvm.unary Log
// CHECK: dvm.store

func.func @test_isfinite(%arg0: tensor<4xf32>) -> tensor<4xi1> attributes {fusion.outlined} {
  %0 = stablehlo.is_finite %arg0 : (tensor<4xf32>) -> tensor<4xi1>
  return %0 : tensor<4xi1>
}
// CHECK-LABEL: func.func @test_isfinite
// CHECK: dvm.load
// CHECK: dvm.unary IsFinite
// CHECK: dvm.store

func.func @test_not(%arg0: tensor<4xi1>) -> tensor<4xi1> attributes {fusion.outlined} {
  %0 = stablehlo.not %arg0 : tensor<4xi1>
  return %0 : tensor<4xi1>
}
// CHECK-LABEL: func.func @test_not
// CHECK: dvm.load
// CHECK: dvm.unary LogicalNot
// CHECK: dvm.store

func.func @test_round(%arg0: tensor<4xf32>) -> tensor<4xf32> attributes {fusion.outlined} {
  %0 = stablehlo.round_nearest_even %arg0 : tensor<4xf32>
  return %0 : tensor<4xf32>
}
// CHECK-LABEL: func.func @test_round
// CHECK: dvm.load
// CHECK: dvm.unary Round
// CHECK: dvm.store

func.func @test_floor(%arg0: tensor<4xf32>) -> tensor<4xf32> attributes {fusion.outlined} {
  %0 = stablehlo.floor %arg0 : tensor<4xf32>
  return %0 : tensor<4xf32>
}
// CHECK-LABEL: func.func @test_floor
// CHECK: dvm.load
// CHECK: dvm.unary Floor
// CHECK: dvm.store

func.func @test_ceil(%arg0: tensor<4xf32>) -> tensor<4xf32> attributes {fusion.outlined} {
  %0 = stablehlo.ceil %arg0 : tensor<4xf32>
  return %0 : tensor<4xf32>
}
// CHECK-LABEL: func.func @test_ceil
// CHECK: dvm.load
// CHECK: dvm.unary Ceil
// CHECK: dvm.store

// Binary Ops

func.func @test_add(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> attributes {fusion.outlined} {
  %0 = stablehlo.add %arg0, %arg1 : tensor<4xf32>
  return %0 : tensor<4xf32>
}
// CHECK-LABEL: func.func @test_add
// CHECK: dvm.load
// CHECK: dvm.load
// CHECK: dvm.binary Add
// CHECK: dvm.store

func.func @test_sub(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> attributes {fusion.outlined} {
  %0 = stablehlo.subtract %arg0, %arg1 : tensor<4xf32>
  return %0 : tensor<4xf32>
}
// CHECK-LABEL: func.func @test_sub
// CHECK: dvm.load
// CHECK: dvm.load
// CHECK: dvm.binary Sub
// CHECK: dvm.store

func.func @test_mul(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> attributes {fusion.outlined} {
  %0 = stablehlo.multiply %arg0, %arg1 : tensor<4xf32>
  return %0 : tensor<4xf32>
}
// CHECK-LABEL: func.func @test_mul
// CHECK: dvm.load
// CHECK: dvm.load
// CHECK: dvm.binary Mul
// CHECK: dvm.store

func.func @test_div(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> attributes {fusion.outlined} {
  %0 = stablehlo.divide %arg0, %arg1 : tensor<4xf32>
  return %0 : tensor<4xf32>
}
// CHECK-LABEL: func.func @test_div
// CHECK: dvm.load
// CHECK: dvm.load
// CHECK: dvm.binary Div
// CHECK: dvm.store

func.func @test_pow(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> attributes {fusion.outlined} {
  %0 = stablehlo.power %arg0, %arg1 : tensor<4xf32>
  return %0 : tensor<4xf32>
}
// CHECK-LABEL: func.func @test_pow
// CHECK: dvm.load
// CHECK: dvm.load
// CHECK: dvm.binary Pow
// CHECK: dvm.store

func.func @test_max(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> attributes {fusion.outlined} {
  %0 = stablehlo.maximum %arg0, %arg1 : tensor<4xf32>
  return %0 : tensor<4xf32>
}
// CHECK-LABEL: func.func @test_max
// CHECK: dvm.load
// CHECK: dvm.load
// CHECK: dvm.binary Maximum
// CHECK: dvm.store

func.func @test_min(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> attributes {fusion.outlined} {
  %0 = stablehlo.minimum %arg0, %arg1 : tensor<4xf32>
  return %0 : tensor<4xf32>
}
// CHECK-LABEL: func.func @test_min
// CHECK: dvm.load
// CHECK: dvm.load
// CHECK: dvm.binary Minimum
// CHECK: dvm.store

func.func @test_and(%arg0: tensor<4xi1>, %arg1: tensor<4xi1>) -> tensor<4xi1> attributes {fusion.outlined} {
  %0 = stablehlo.and %arg0, %arg1 : tensor<4xi1>
  return %0 : tensor<4xi1>
}
// CHECK-LABEL: func.func @test_and
// CHECK: dvm.load
// CHECK: dvm.load
// CHECK: dvm.binary LogicalAnd
// CHECK: dvm.store

func.func @test_or(%arg0: tensor<4xi1>, %arg1: tensor<4xi1>) -> tensor<4xi1> attributes {fusion.outlined} {
  %0 = stablehlo.or %arg0, %arg1 : tensor<4xi1>
  return %0 : tensor<4xi1>
}
// CHECK-LABEL: func.func @test_or
// CHECK: dvm.load
// CHECK: dvm.load
// CHECK: dvm.binary LogicalOr
// CHECK: dvm.store

func.func @test_eq(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xi1> attributes {fusion.outlined} {
  %0 = stablehlo.compare  EQ, %arg0, %arg1,  FLOAT : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
  return %0 : tensor<4xi1>
}
// CHECK-LABEL: func.func @test_eq
// CHECK: dvm.load
// CHECK: dvm.load
// CHECK: dvm.binary Equal
// CHECK: dvm.store

func.func @test_ne(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xi1> attributes {fusion.outlined} {
  %0 = stablehlo.compare  NE, %arg0, %arg1,  FLOAT : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
  return %0 : tensor<4xi1>
}
// CHECK-LABEL: func.func @test_ne
// CHECK: dvm.load
// CHECK: dvm.load
// CHECK: dvm.binary NotEqual
// CHECK: dvm.store

func.func @test_gt(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xi1> attributes {fusion.outlined} {
  %0 = stablehlo.compare  GT, %arg0, %arg1,  FLOAT : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
  return %0 : tensor<4xi1>
}
// CHECK-LABEL: func.func @test_gt
// CHECK: dvm.load
// CHECK: dvm.load
// CHECK: dvm.binary Greater
// CHECK: dvm.store

func.func @test_ge(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xi1> attributes {fusion.outlined} {
  %0 = stablehlo.compare  GE, %arg0, %arg1,  FLOAT : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
  return %0 : tensor<4xi1>
}
// CHECK-LABEL: func.func @test_ge
// CHECK: dvm.load
// CHECK: dvm.load
// CHECK: dvm.binary GreaterEqual
// CHECK: dvm.store

func.func @test_lt(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xi1> attributes {fusion.outlined} {
  %0 = stablehlo.compare  LT, %arg0, %arg1,  FLOAT : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
  return %0 : tensor<4xi1>
}
// CHECK-LABEL: func.func @test_lt
// CHECK: dvm.load
// CHECK: dvm.load
// CHECK: dvm.binary Less
// CHECK: dvm.store

func.func @test_le(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xi1> attributes {fusion.outlined} {
  %0 = stablehlo.compare  LE, %arg0, %arg1,  FLOAT : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
  return %0 : tensor<4xi1>
}
// CHECK-LABEL: func.func @test_le
// CHECK: dvm.load
// CHECK: dvm.load
// CHECK: dvm.binary LessEqual
// CHECK: dvm.store

// Dot Ops

func.func @test_dot_2d(%arg0: tensor<2x3xf32>, %arg1: tensor<3x2xf32>) -> tensor<2x2xf32> attributes {fusion.outlined} {
  %0 = stablehlo.dot %arg0, %arg1 : (tensor<2x3xf32>, tensor<3x2xf32>) -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}
// CHECK-LABEL: func.func @test_dot_2d
// CHECK: dvm.load
// CHECK: dvm.load
// CHECK: dvm.matmul %{{.*}}, %{{.*}} trans_a false trans_b false
// CHECK: dvm.store

func.func @test_dot_vector_vector(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<f32> attributes {fusion.outlined} {
  %0 = stablehlo.dot %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<f32>
  return %0 : tensor<f32>
}
// CHECK-LABEL: func.func @test_dot_vector_vector
// CHECK: dvm.load
// CHECK: dvm.load
// CHECK: dvm.reshape
// CHECK: dvm.reshape
// CHECK: dvm.matmul
// CHECK: dvm.reshape
// CHECK: dvm.store

func.func @test_dot_matrix_vector(%arg0: tensor<2x3xf32>, %arg1: tensor<3xf32>) -> tensor<2xf32> attributes {fusion.outlined} {
  %0 = stablehlo.dot %arg0, %arg1 : (tensor<2x3xf32>, tensor<3xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}
// CHECK-LABEL: func.func @test_dot_matrix_vector
// CHECK: dvm.load
// CHECK: dvm.load
// CHECK: dvm.reshape
// CHECK: dvm.matmul
// CHECK: dvm.reshape
// CHECK: dvm.store

func.func @test_dot_vector_matrix(%arg0: tensor<2xf32>, %arg1: tensor<2x3xf32>) -> tensor<3xf32> attributes {fusion.outlined} {
  %0 = stablehlo.dot %arg0, %arg1 : (tensor<2xf32>, tensor<2x3xf32>) -> tensor<3xf32>
  return %0 : tensor<3xf32>
}
// CHECK-LABEL: func.func @test_dot_vector_matrix
// CHECK: dvm.load
// CHECK: dvm.load
// CHECK: dvm.reshape
// CHECK: dvm.matmul
// CHECK: dvm.reshape
// CHECK: dvm.store

func.func @test_dot_general_standard(%arg0: tensor<1x2x3xf32>, %arg1: tensor<1x3x2xf32>) -> tensor<1x2x2xf32> attributes {fusion.outlined} {
  %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<1x2x3xf32>, tensor<1x3x2xf32>) -> tensor<1x2x2xf32>
  return %0 : tensor<1x2x2xf32>
}
// CHECK-LABEL: func.func @test_dot_general_standard
// CHECK: dvm.load
// CHECK: dvm.load
// CHECK: dvm.matmul %{{.*}}, %{{.*}} trans_a false trans_b false
// CHECK: dvm.store

func.func @test_dot_general_transposed(%arg0: tensor<1x3x2xf32>, %arg1: tensor<1x2x3xf32>) -> tensor<1x2x2xf32> attributes {fusion.outlined} {
  %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [1] x [2] : (tensor<1x3x2xf32>, tensor<1x2x3xf32>) -> tensor<1x2x2xf32>
  return %0 : tensor<1x2x2xf32>
}
// CHECK-LABEL: func.func @test_dot_general_transposed
// CHECK: dvm.load
// CHECK: dvm.load
// CHECK: dvm.matmul %{{.*}}, %{{.*}} trans_a true trans_b true
// CHECK: dvm.store

func.func @test_dot_general_multi_dim(%arg0: tensor<1x4x2x3xf32>, %arg1: tensor<1x4x3x2xf32>) -> tensor<1x4x2x2xf32> attributes {fusion.outlined} {
  // Batch dims: [0, 1]
  // Contracting dims: LHS [3], RHS [2] -> K=3
  // Non-contracting dims: LHS [2], RHS [3] -> M=2, N=2
  %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2] : (tensor<1x4x2x3xf32>, tensor<1x4x3x2xf32>) -> tensor<1x4x2x2xf32>
  return %0 : tensor<1x4x2x2xf32>
}
// CHECK-LABEL: func.func @test_dot_general_multi_dim
// CHECK: dvm.load
// CHECK: dvm.load
// CHECK: dvm.matmul %{{.*}}, %{{.*}} trans_a false trans_b false
// CHECK: dvm.store

func.func @test_dot_general_flatten_M(%arg0: tensor<1x2x3x4xf32>, %arg1: tensor<1x4x5xf32>) -> tensor<1x2x3x5xf32> attributes {fusion.outlined} {
  // Batch dims: [0]
  // Contracting dims: LHS [3], RHS [1] -> K=4
  // Non-contracting dims: LHS [1, 2], RHS [2] -> M=6, N=5
  // Requires flatten M dims [1, 2]
  %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [3] x [1] : (tensor<1x2x3x4xf32>, tensor<1x4x5xf32>) -> tensor<1x2x3x5xf32>
  return %0 : tensor<1x2x3x5xf32>
}
// CHECK-LABEL: func.func @test_dot_general_flatten_M
// CHECK: dvm.load
// CHECK: dvm.load
// CHECK: dvm.reshape
// CHECK: dvm.matmul
// CHECK: dvm.reshape
// CHECK: dvm.store

func.func @test_dot_general_flatten_N(%arg0: tensor<1x2x3xf32>, %arg1: tensor<1x3x4x5xf32>) -> tensor<1x2x4x5xf32> attributes {fusion.outlined} {
  // Batch dims: [0]
  // Contracting dims: LHS [2], RHS [1] -> K=3
  // Non-contracting dims: LHS [1], RHS [2, 3] -> M=2, N=20
  // Requires flatten N dims [2, 3]
  %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<1x2x3xf32>, tensor<1x3x4x5xf32>) -> tensor<1x2x4x5xf32>
  return %0 : tensor<1x2x4x5xf32>
}
// CHECK-LABEL: func.func @test_dot_general_flatten_N
// CHECK: dvm.load
// CHECK: dvm.load
// CHECK: dvm.reshape
// CHECK: dvm.matmul
// CHECK: dvm.reshape
// CHECK: dvm.store

func.func @test_dot_general_flatten_K(%arg0: tensor<1x2x3x4xf32>, %arg1: tensor<1x3x4x5xf32>) -> tensor<1x2x5xf32> attributes {fusion.outlined} {
  // Batch dims: [0]
  // Contracting dims: LHS [2, 3], RHS [1, 2] -> K=12
  // Non-contracting dims: LHS [1], RHS [3] -> M=2, N=5
  // Requires flatten K dims
  %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2, 3] x [1, 2] : (tensor<1x2x3x4xf32>, tensor<1x3x4x5xf32>) -> tensor<1x2x5xf32>
  return %0 : tensor<1x2x5xf32>
}
// CHECK-LABEL: func.func @test_dot_general_flatten_K
// CHECK: dvm.load
// CHECK: dvm.load
// CHECK: dvm.reshape
// CHECK: dvm.reshape
// CHECK: dvm.matmul
// CHECK: dvm.store

func.func @test_dot_general_transposed_flatten(%arg0: tensor<1x4x2x3xf32>, %arg1: tensor<1x2x3x5xf32>) -> tensor<1x4x5xf32> attributes {fusion.outlined} {
  // Batch dims: [0]
  // Contracting dims: LHS [2, 3], RHS [1, 2] -> K=6
  // Non-contracting dims: LHS [1], RHS [3] -> M=4, N=5
  // LHS: [B, M, K1, K2] -> Transposed K? No, standard K is at end.
  // Wait, standard LHS is [B, M, K].
  // Here LHS is [B, M, K1, K2]. After flatten K, it is [B, M, K]. So standard.
  // RHS: [B, K1, K2, N]. After flatten K, it is [B, K, N]. So standard.
  %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2, 3] x [1, 2] : (tensor<1x4x2x3xf32>, tensor<1x2x3x5xf32>) -> tensor<1x4x5xf32>
  return %0 : tensor<1x4x5xf32>
}
// CHECK-LABEL: func.func @test_dot_general_transposed_flatten
// CHECK: dvm.load
// CHECK: dvm.load
// CHECK: dvm.reshape
// CHECK: dvm.reshape
// CHECK: dvm.matmul
// CHECK: dvm.store

// Constant Lifting Tests

func.func @test_lift_constant_callee(%arg0: tensor<4xf32>) -> tensor<4xf32> attributes {fusion.outlined} {
  %cst = stablehlo.constant dense<1.0> : tensor<4xf32>
  %0 = stablehlo.add %arg0, %cst : tensor<4xf32>
  return %0 : tensor<4xf32>
}
// CHECK-LABEL: func.func @test_lift_constant_callee
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4xf32>, %[[ARG1:.*]]: tensor<4xf32>)
// CHECK: %[[LOAD_ARG0:.*]] = dvm.load %[[ARG0]]
// CHECK: %[[LOAD_ARG1:.*]] = dvm.load %[[ARG1]]
// CHECK: dvm.binary Add %[[LOAD_ARG0]], %[[LOAD_ARG1]]
// CHECK-NOT: stablehlo.constant

func.func @test_lift_constant_caller(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %0 = func.call @test_lift_constant_callee(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  %1 = stablehlo.add %0, %arg0 : tensor<4xf32>
  return %1 : tensor<4xf32>
}
// CHECK-LABEL: func.func @test_lift_constant_caller
// CHECK: %[[CST:.*]] = stablehlo.constant dense<1.000000e+00> : tensor<4xf32>
// CHECK: %[[CALL_RES:.*]] = call @test_lift_constant_callee(%{{.*}}, %[[CST]])
// CHECK: stablehlo.add %[[CALL_RES]], %{{.*}}
// CHECK-NOT: dvm.binary Add
// CHECK-NOT: dvm.load
// CHECK-NOT: dvm.store
