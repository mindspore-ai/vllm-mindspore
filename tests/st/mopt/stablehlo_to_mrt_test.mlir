// RUN: mopt-opt --convert-stablehlo-to-mrt --mlir-print-ir-before-all --mlir-print-ir-after-all %s

// Test reshape conversion
// CHECK-LABEL: func.func @test_reshape
func.func @test_reshape(%arg0: tensor<2x3x4xf32>) -> tensor<24xf32> {
  // CHECK: %[[SHAPE:.*]] = arith.constant dense<24> : tensor<1xi64>
  // CHECK: %[[RESULT:.*]] = mrt.reshape %arg0, %[[SHAPE]] : (tensor<2x3x4xf32>, tensor<1xi64>) -> tensor<24xf32>
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

// Test dot (matmul) conversion
// CHECK-LABEL: func.func @test_dot
func.func @test_dot(%arg0: tensor<2x3xf32>, %arg1: tensor<3x4xf32>) -> tensor<2x4xf32> {
  // CHECK: %[[RESULT:.*]] = mrt.matmul %arg0, %arg1 : (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>
  %0 = stablehlo.dot %arg0, %arg1 : (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

// Test sigmoid (logistic) conversion
// CHECK-LABEL: func.func @test_sigmoid
func.func @test_sigmoid(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
  // CHECK: %[[RESULT:.*]] = mrt.sigmoid %arg0 : (tensor<2x3xf32>) -> tensor<2x3xf32>
  %0 = stablehlo.logistic %arg0 : tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
}

// Test ReLU pattern (max with zero)
// CHECK-LABEL: func.func @test_relu
func.func @test_relu(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %zero = stablehlo.constant dense<0.0> : tensor<2x3xf32>
  // CHECK: %[[RESULT:.*]] = mrt.relu %arg0 : (tensor<2x3xf32>) -> tensor<2x3xf32>
  %0 = stablehlo.maximum %arg0, %zero : tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
}

// Test concatenate conversion
// CHECK-LABEL: func.func @test_concat
func.func @test_concat(%arg0: tensor<2x3xf32>, %arg1: tensor<2x4xf32>) -> tensor<2x7xf32> {
  // CHECK: %[[RESULT:.*]] = mrt.concat %arg0, %arg1 along axis 1 : (tensor<2x3xf32>, tensor<2x4xf32>) -> tensor<2x7xf32>
  %0 = stablehlo.concatenate %arg0, %arg1, dim = 1 : (tensor<2x3xf32>, tensor<2x4xf32>) -> tensor<2x7xf32>
  return %0 : tensor<2x7xf32>
}

// Test batch norm conversion
// CHECK-LABEL: func.func @test_batch_norm
func.func @test_batch_norm(%arg0: tensor<2x3x4x5xf32>, %scale: tensor<3xf32>, 
                            %offset: tensor<3xf32>, %mean: tensor<3xf32>, 
                            %variance: tensor<3xf32>) -> tensor<2x3x4x5xf32> {
  // CHECK: %[[RESULT:.*]] = mrt.batch_norm %arg0, %scale, %offset, %mean, %variance 
  // CHECK-SAME: {epsilon = 1.000000e-05 : f32, is_training = false}
  %0 = "stablehlo.batch_norm_inference"(%arg0, %scale, %offset, %mean, %variance) {
       epsilon = 1.000000e-05 : f32, 
       feature_index = 1 : i64
       } : (tensor<2x3x4x5xf32>, tensor<3xf32>, tensor<3xf32>, tensor<3xf32>, tensor<3xf32>) -> tensor<2x3x4x5xf32>
  return %0 : tensor<2x3x4x5xf32>
}

// Test complex pipeline with multiple operations
// CHECK-LABEL: func.func @test_pipeline
func.func @test_pipeline(%arg0: tensor<2x3xf32>, %arg1: tensor<2x3xf32>) -> tensor<2x3xf32> {
  // CHECK: %[[MUL:.*]] = mrt.mul %arg0, %arg1
  %0 = stablehlo.multiply %arg0, %arg1 : tensor<2x3xf32>
  
  // CHECK: %[[SIGMOID:.*]] = mrt.sigmoid %[[MUL]]
  %1 = stablehlo.logistic %0 : tensor<2x3xf32>
  
  // CHECK: %[[SHAPE:.*]] = arith.constant dense<[6]> : tensor<1xi64>
  // CHECK: %[[RESHAPE:.*]] = mrt.reshape %[[SIGMOID]], %[[SHAPE]]
  %2 = stablehlo.reshape %1 : (tensor<2x3xf32>) -> tensor<6xf32>
  
  // CHECK: %[[SHAPE2:.*]] = arith.constant dense<[2, 3]> : tensor<2xi64>
  // CHECK: %[[RESULT:.*]] = mrt.reshape %[[RESHAPE]], %[[SHAPE2]]
  %3 = stablehlo.reshape %2 : (tensor<6xf32>) -> tensor<2x3xf32>
  
  return %3 : tensor<2x3xf32>
}

