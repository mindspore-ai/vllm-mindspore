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
  // CHECK: %[[AXIS:.*]] = mrt.constant.i64
  // CHECK: %[[RESULT:.*]] = mrt.concat %arg0, %arg1, %[[AXIS]]
  %0 = stablehlo.concatenate %arg0, %arg1, dim = 1 : (tensor<2x3xf32>, tensor<2x4xf32>) -> tensor<2x7xf32>
  return %0 : tensor<2x7xf32>
}

// Test batch norm conversion
// CHECK-LABEL: func.func @test_batch_norm
func.func @test_batch_norm(%arg0: tensor<2x3x4x5xf32>, %scale: tensor<3xf32>, 
                            %offset: tensor<3xf32>, %mean: tensor<3xf32>, 
                            %variance: tensor<3xf32>) -> tensor<2x3x4x5xf32> {
  // CHECK: %[[EPSILON:.*]] = mrt.constant.f32
  // CHECK: %[[IS_TRAINING:.*]] = mrt.constant.boolean
  // CHECK: %[[RESULT:.*]] = mrt.batch_norm %arg0, %scale, %offset, %mean, %variance, %[[EPSILON]], %[[IS_TRAINING]]
  %0 = "stablehlo.batch_norm_inference"(%arg0, %scale, %offset, %mean, %variance) {
       epsilon = 1.000000e-05 : f32, 
       feature_index = 1 : i64
       } : (tensor<2x3x4x5xf32>, tensor<3xf32>, tensor<3xf32>, tensor<3xf32>, tensor<3xf32>) -> tensor<2x3x4x5xf32>
  return %0 : tensor<2x3x4x5xf32>
}

// Test convolution conversion
// CHECK-LABEL: func.func @test_conv
func.func @test_conv(%arg0: tensor<1x3x4x4xf32>, %arg1: tensor<2x3x2x2xf32>) -> tensor<1x2x3x3xf32> {
  // CHECK: %[[STRIDES:.*]] = mrt.constant.i64_array
  // CHECK: %[[PADDING:.*]] = mrt.constant.i64_array
  // CHECK: %[[DILATION:.*]] = mrt.constant.i64_array
  // CHECK: %[[HAS_BIAS:.*]] = mrt.constant.boolean
  // CHECK: %[[RESULT:.*]] = mrt.conv %arg0, %arg1, %{{.*}}, %[[STRIDES]], %[[PADDING]], %[[DILATION]], %[[HAS_BIAS]]
  %0 = stablehlo.convolution(%arg0, %arg1) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [0, 0]} {feature_group_count = 1 : i64, batch_group_count = 1 : i64, precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]} : (tensor<1x3x4x4xf32>, tensor<2x3x2x2xf32>) -> tensor<1x2x3x3xf32>
  return %0 : tensor<1x2x3x3xf32>
}

// Test softmax conversion
// Note: This test uses a simplified softmax pattern. The actual conversion
// may require implementing a pattern matcher for the full softmax computation.
// CHECK-LABEL: func.func @test_softmax
func.func @test_softmax(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
  // CHECK: %[[AXIS:.*]] = mrt.constant.i64
  // CHECK: %[[RESULT:.*]] = mrt.softmax %arg0, %[[AXIS]]
  // For now, using a placeholder that represents softmax computation
  // The actual conversion pattern needs to be implemented to match
  // the full softmax computation (reduce-max, subtract, exp, reduce-sum, divide)
  %max_init = stablehlo.constant dense<-3.40282347E+38> : tensor<f32>
  %0 = "stablehlo.reduce"(%arg0, %max_init) ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %1 = stablehlo.maximum %arg1, %arg2 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) {
      dimensions = array<i64: 1>
    } : (tensor<2x3xf32>, tensor<f32>) -> tensor<2xf32>
  %1 = stablehlo.broadcast_in_dim %0, dims = [0] : (tensor<2xf32>) -> tensor<2x3xf32>
  %2 = stablehlo.subtract %arg0, %1 : tensor<2x3xf32>
  %3 = stablehlo.exponential %2 : tensor<2x3xf32>
  %sum_init = stablehlo.constant dense<0.0> : tensor<f32>
  %4 = "stablehlo.reduce"(%3, %sum_init) ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %5 = stablehlo.add %arg1, %arg2 : tensor<f32>
      stablehlo.return %5 : tensor<f32>
    }) {
      dimensions = array<i64: 1>
    } : (tensor<2x3xf32>, tensor<f32>) -> tensor<2xf32>
  %5 = stablehlo.broadcast_in_dim %4, dims = [0] : (tensor<2xf32>) -> tensor<2x3xf32>
  %6 = stablehlo.divide %3, %5 : tensor<2x3xf32>
  return %6 : tensor<2x3xf32>
}

// Test pooling (max pool) conversion
// CHECK-LABEL: func.func @test_pool
func.func @test_pool(%arg0: tensor<1x3x4x4xf32>) -> tensor<1x3x2x2xf32> {
  // CHECK: %[[KERNEL_SIZE:.*]] = mrt.constant.i64_array
  // CHECK: %[[STRIDES:.*]] = mrt.constant.i64_array
  // CHECK: %[[PADDING:.*]] = mrt.constant.i64_array
  // CHECK: %[[POOL_TYPE:.*]] = mrt.constant.string
  // CHECK: %[[RESULT:.*]] = mrt.pool %arg0, %[[KERNEL_SIZE]], %[[STRIDES]], %[[PADDING]], %[[POOL_TYPE]]
  %init_value = stablehlo.constant dense<-3.40282347E+38> : tensor<f32>
  %0 = "stablehlo.reduce_window"(%arg0, %init_value) ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %1 = stablehlo.maximum %arg1, %arg2 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) {
      window_dimensions = array<i64: 1, 1, 2, 2>,
      window_strides = array<i64: 1, 1, 2, 2>,
      base_dilations = array<i64: 1, 1, 1, 1>,
      window_dilations = array<i64: 1, 1, 1, 1>,
      padding = dense<[[0, 0], [0, 0], [0, 0], [0, 0]]> : tensor<4x2xi64>
    } : (tensor<1x3x4x4xf32>, tensor<f32>) -> tensor<1x3x2x2xf32>
  return %0 : tensor<1x3x2x2xf32>
}

// Test split conversion
// CHECK-LABEL: func.func @test_split
func.func @test_split(%arg0: tensor<2x6xf32>) -> (tensor<2x3xf32>, tensor<2x3xf32>) {
  // CHECK: %[[AXIS:.*]] = mrt.constant.i64
  // CHECK: %[[SPLIT_SIZES:.*]] = mrt.constant.i64_array
  // CHECK: %[[RESULT0:.*]], %[[RESULT1:.*]] = mrt.split %arg0, %[[AXIS]], %[[SPLIT_SIZES]]
  %0 = stablehlo.slice %arg0 [0:2, 0:3] : (tensor<2x6xf32>) -> tensor<2x3xf32>
  %1 = stablehlo.slice %arg0 [0:2, 3:6] : (tensor<2x6xf32>) -> tensor<2x3xf32>
  return %0, %1 : tensor<2x3xf32>, tensor<2x3xf32>
}

// Test complex pipeline with multiple operations
// CHECK-LABEL: func.func @test_pipeline
func.func @test_pipeline(%arg0: tensor<2x3xf32>, %arg1: tensor<2x3xf32>) -> tensor<2x3xf32> {
  // CHECK: %[[MUL:.*]] = mrt.mul %arg0, %arg1
  %0 = stablehlo.multiply %arg0, %arg1 : tensor<2x3xf32>
  
  // CHECK: %[[SIGMOID:.*]] = mrt.sigmoid %[[MUL]]
  %1 = stablehlo.logistic %0 : tensor<2x3xf32>
  
  // CHECK: %[[SHAPE:.*]] = mrt.constant.i64_array
  // CHECK: %[[RESHAPE:.*]] = mrt.reshape %[[SIGMOID]], %[[SHAPE]]
  %2 = stablehlo.reshape %1 : (tensor<2x3xf32>) -> tensor<6xf32>
  
  // CHECK: %[[SHAPE2:.*]] = mrt.constant.i64_array
  // CHECK: %[[RESULT:.*]] = mrt.reshape %[[RESHAPE]], %[[SHAPE2]]
  %3 = stablehlo.reshape %2 : (tensor<6xf32>) -> tensor<2x3xf32>
  
  return %3 : tensor<2x3xf32>
}

