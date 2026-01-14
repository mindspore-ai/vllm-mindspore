// RUN: mopt-opt --convert-stablehlo-to-mrt --mlir-print-ir-before-all --mlir-print-ir-after-all %s

// Test reshape conversion
// CHECK-LABEL: func.func @test_reshape
func.func @test_reshape(%arg0: tensor<2x3x4xf32>) -> tensor<24xf32> {
  // CHECK: %[[SHAPE:.*]] = mrt.constant.i64_array
  // CHECK: %[[RESULT:.*]] = mrt.reshape %arg0, %[[SHAPE]]
  %0 = stablehlo.reshape %arg0 : (tensor<2x3x4xf32>) -> tensor<24xf32>
  return %0 : tensor<24xf32>
}

// Test multiply conversion
// CHECK-LABEL: func.func @test_multiply
func.func @test_multiply(%arg0: tensor<2x3xf32>, %arg1: tensor<2x3xf32>) -> tensor<2x3xf32> {
  // CHECK: %[[RESULT:.*]] = mrt.mul %arg0, %arg1 : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  %0 = stablehlo.multiply %arg0, %arg1 : tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
}

// Test add conversion (alpha=1.0)
// CHECK-LABEL: func.func @test_add
func.func @test_add(%arg0: tensor<2x3xf32>, %arg1: tensor<2x3xf32>) -> tensor<2x3xf32> {
  // CHECK: %[[ALPHA:.*]] = mrt.constant.f64 1.000000e+00 : !mrt.f64
  // CHECK: %[[RESULT:.*]] = mrt.add %arg0, %arg1, %[[ALPHA]] : (tensor<2x3xf32>, tensor<2x3xf32>, !mrt.f64) -> tensor<2x3xf32>
  %0 = stablehlo.add %arg0, %arg1 : tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
}

// Test subtract conversion (alpha=1.0)
// CHECK-LABEL: func.func @test_subtract
func.func @test_subtract(%arg0: tensor<2x3xf32>, %arg1: tensor<2x3xf32>) -> tensor<2x3xf32> {
  // CHECK: %[[ALPHA:.*]] = mrt.constant.f64 1.000000e+00 : !mrt.f64
  // CHECK: %[[RESULT:.*]] = mrt.sub %arg0, %arg1, %[[ALPHA]] : (tensor<2x3xf32>, tensor<2x3xf32>, !mrt.f64) -> tensor<2x3xf32>
  %0 = stablehlo.subtract %arg0, %arg1 : tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
}
