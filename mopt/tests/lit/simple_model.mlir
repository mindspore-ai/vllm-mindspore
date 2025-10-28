// RUN: mrt-opt %s | FileCheck %s

// CHECK-LABEL: @test_matmul
func.func @test_matmul(%lhs: tensor<10x20xf32>, %rhs: tensor<20x30xf32>) -> tensor<10x30xf32> {
  // CHECK: mrt.matmul
  %0 = mrt.matmul %lhs, %rhs {transpose_a = false, transpose_b = false, has_bias = false} : (tensor<10x20xf32>, tensor<20x30xf32>) -> tensor<10x30xf32>
  return %0 : tensor<10x30xf32>
}
