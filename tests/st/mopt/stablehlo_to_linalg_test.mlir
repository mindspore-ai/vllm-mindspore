// RUN: mopt-opt --stablehlo-legalize-to-linalg %s | FileCheck %s

// Test element-wise add conversion
// CHECK-LABEL: func.func @test_add
func.func @test_add(%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>) -> tensor<4x8xf32> {
  // CHECK: linalg.generic
  // CHECK: arith.addf
  %0 = stablehlo.add %arg0, %arg1 : tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// Test element-wise multiply conversion
// CHECK-LABEL: func.func @test_multiply
func.func @test_multiply(%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>) -> tensor<4x8xf32> {
  // CHECK: linalg.generic
  // CHECK: arith.mulf
  %0 = stablehlo.multiply %arg0, %arg1 : tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// Test element-wise subtract conversion
// CHECK-LABEL: func.func @test_subtract
func.func @test_subtract(%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>) -> tensor<4x8xf32> {
  // CHECK: linalg.generic
  // CHECK: arith.subf
  %0 = stablehlo.subtract %arg0, %arg1 : tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// Test element-wise divide conversion
// CHECK-LABEL: func.func @test_divide
func.func @test_divide(%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>) -> tensor<4x8xf32> {
  // CHECK: linalg.generic
  // CHECK: arith.divf
  %0 = stablehlo.divide %arg0, %arg1 : tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// Test element-wise max conversion
// CHECK-LABEL: func.func @test_maximum
func.func @test_maximum(%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>) -> tensor<4x8xf32> {
  // CHECK: linalg.generic
  // CHECK: arith.maximumf
  %0 = stablehlo.maximum %arg0, %arg1 : tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// Test element-wise min conversion
// CHECK-LABEL: func.func @test_minimum
func.func @test_minimum(%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>) -> tensor<4x8xf32> {
  // CHECK: linalg.generic
  // CHECK: arith.minimumf
  %0 = stablehlo.minimum %arg0, %arg1 : tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// Test abs conversion
// CHECK-LABEL: func.func @test_abs
func.func @test_abs(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> {
  // CHECK: linalg.generic
  // CHECK: math.absf
  %0 = stablehlo.abs %arg0 : tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// Test exponential conversion
// CHECK-LABEL: func.func @test_exp
func.func @test_exp(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> {
  // CHECK: linalg.generic
  // CHECK: math.exp
  %0 = stablehlo.exponential %arg0 : tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// Test logarithm conversion
// CHECK-LABEL: func.func @test_log
func.func @test_log(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> {
  // CHECK: linalg.generic
  // CHECK: math.log
  %0 = stablehlo.log %arg0 : tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// Test sqrt conversion
// CHECK-LABEL: func.func @test_sqrt
func.func @test_sqrt(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> {
  // CHECK: linalg.generic
  // CHECK: math.sqrt
  %0 = stablehlo.sqrt %arg0 : tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// Test tanh conversion
// CHECK-LABEL: func.func @test_tanh
func.func @test_tanh(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> {
  // CHECK: linalg.generic
  // CHECK: math.tanh
  %0 = stablehlo.tanh %arg0 : tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// Test negate conversion
// CHECK-LABEL: func.func @test_negate
func.func @test_negate(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> {
  // CHECK: linalg.generic
  // CHECK: arith.negf
  %0 = stablehlo.negate %arg0 : tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// Test dot (matrix multiplication) conversion
// CHECK-LABEL: func.func @test_dot
func.func @test_dot(%arg0: tensor<4x8xf32>, %arg1: tensor<8x16xf32>) -> tensor<4x16xf32> {
  // CHECK: linalg.matmul
  %0 = stablehlo.dot %arg0, %arg1 : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>
  return %0 : tensor<4x16xf32>
}

// Test dot_general (batch matmul) conversion
// CHECK-LABEL: func.func @test_dot_general
func.func @test_dot_general(%arg0: tensor<2x4x8xf32>, %arg1: tensor<2x8x16xf32>) -> tensor<2x4x16xf32> {
  // CHECK: linalg.batch_matmul
  %0 = "stablehlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [2],
      rhs_contracting_dimensions = [1]
    >
  } : (tensor<2x4x8xf32>, tensor<2x8x16xf32>) -> tensor<2x4x16xf32>
  return %0 : tensor<2x4x16xf32>
}

// Test reduce (sum) conversion
// CHECK-LABEL: func.func @test_reduce_sum
func.func @test_reduce_sum(%arg0: tensor<4x8xf32>) -> tensor<4xf32> {
  // CHECK: linalg.reduce
  %init = stablehlo.constant dense<0.0> : tensor<f32>
  %0 = "stablehlo.reduce"(%arg0, %init) ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %1 = stablehlo.add %arg1, %arg2 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) {
      dimensions = array<i64: 1>
    } : (tensor<4x8xf32>, tensor<f32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// Test reduce (max) conversion
// CHECK-LABEL: func.func @test_reduce_max
func.func @test_reduce_max(%arg0: tensor<4x8xf32>) -> tensor<4xf32> {
  // CHECK: linalg.reduce
  %init = stablehlo.constant dense<-3.40282347E+38> : tensor<f32>
  %0 = "stablehlo.reduce"(%arg0, %init) ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %1 = stablehlo.maximum %arg1, %arg2 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) {
      dimensions = array<i64: 1>
    } : (tensor<4x8xf32>, tensor<f32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// Test broadcast_in_dim conversion
// CHECK-LABEL: func.func @test_broadcast
func.func @test_broadcast(%arg0: tensor<4xf32>) -> tensor<4x8xf32> {
  // CHECK: linalg.generic
  %0 = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<4xf32>) -> tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// Test transpose conversion
// CHECK-LABEL: func.func @test_transpose
func.func @test_transpose(%arg0: tensor<4x8x16xf32>) -> tensor<16x4x8xf32> {
  // CHECK: linalg.transpose
  %0 = stablehlo.transpose %arg0, dims = [2, 0, 1] : (tensor<4x8x16xf32>) -> tensor<16x4x8xf32>
  return %0 : tensor<16x4x8xf32>
}

// Test concatenate conversion
// CHECK-LABEL: func.func @test_concatenate
func.func @test_concatenate(%arg0: tensor<4x8xf32>, %arg1: tensor<4x16xf32>) -> tensor<4x24xf32> {
  // CHECK: tensor.concat
  %0 = "stablehlo.concatenate"(%arg0, %arg1) {dimension = 1 : i64} : (tensor<4x8xf32>, tensor<4x16xf32>) -> tensor<4x24xf32>
  return %0 : tensor<4x24xf32>
}

// Test convolution conversion (NHWC format for canonical conversion)
// CHECK-LABEL: func.func @test_convolution
func.func @test_convolution(%arg0: tensor<1x224x224x3xf32>, %arg1: tensor<7x7x3x64xf32>) -> tensor<1x112x112x64xf32> {
  // CHECK: linalg.conv_2d_nhwc_hwcf
  %0 = stablehlo.convolution(%arg0, %arg1)
    dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
    window = {stride = [2, 2], pad = [[3, 3], [3, 3]], rhs_dilate = [1, 1]}
    {feature_group_count = 1 : i64, batch_group_count = 1 : i64}
    : (tensor<1x224x224x3xf32>, tensor<7x7x3x64xf32>) -> tensor<1x112x112x64xf32>
  return %0 : tensor<1x112x112x64xf32>
}

// Test complex pipeline with multiple operations
// CHECK-LABEL: func.func @test_pipeline
func.func @test_pipeline(%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>) -> tensor<4x8xf32> {
  // Element-wise multiply
  // CHECK: linalg.generic
  // CHECK: arith.mulf
  %0 = stablehlo.multiply %arg0, %arg1 : tensor<4x8xf32>

  // Element-wise add
  // CHECK: linalg.generic
  // CHECK: arith.addf
  %1 = stablehlo.add %0, %arg0 : tensor<4x8xf32>

  // Exponential
  // CHECK: linalg.generic
  // CHECK: math.exp
  %2 = stablehlo.exponential %1 : tensor<4x8xf32>

  // Tanh
  // CHECK: linalg.generic
  // CHECK: math.tanh
  %3 = stablehlo.tanh %2 : tensor<4x8xf32>

  return %3 : tensor<4x8xf32>
}

// Test dynamic shapes
// CHECK-LABEL: func.func @test_dynamic_shapes
func.func @test_dynamic_shapes(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
  // CHECK: linalg.generic
  // CHECK: arith.addf
  %0 = stablehlo.add %arg0, %arg1 : tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
