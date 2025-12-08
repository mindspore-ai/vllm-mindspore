// RUN: mopt-opt %s

// Test rms_norm operation
// CHECK-LABEL: func.func @test_rms_norm
func.func @test_rms_norm(%arg0: tensor<2x4xf32>, %arg1: tensor<4xf32>) -> (tensor<2x4xf32>, tensor<2xf32>) {
  %epsilon = mrt.constant.f64 1.000000e-05 : !mrt.f64
  %0, %1 = mrt.rms_norm %arg0, %arg1, %epsilon : (tensor<2x4xf32>, tensor<4xf32>, !mrt.f64) -> (tensor<2x4xf32>, tensor<2xf32>)
  return %0, %1 : tensor<2x4xf32>, tensor<2xf32>
}

// Test add_rms_norm operation
// CHECK-LABEL: func.func @test_add_rms_norm
func.func @test_add_rms_norm(%arg0: tensor<2x4xf32>, %arg1: tensor<2x4xf32>, %arg2: tensor<4xf32>) -> (tensor<2x4xf32>, tensor<2xf32>, tensor<2x4xf32>) {
  %epsilon = mrt.constant.f64 1.000000e-05 : !mrt.f64
  %0, %1, %2 = mrt.add_rms_norm %arg0, %arg1, %arg2, %epsilon : (tensor<2x4xf32>, tensor<2x4xf32>, tensor<4xf32>, !mrt.f64) -> (tensor<2x4xf32>, tensor<2xf32>, tensor<2x4xf32>)
  return %0, %1, %2 : tensor<2x4xf32>, tensor<2xf32>, tensor<2x4xf32>
}

// Test moe_init_routing_v3 operation
// CHECK-LABEL: func.func @test_moe_init_routing_v3
func.func @test_moe_init_routing_v3(%arg0: tensor<8x128xf32>, %arg1: tensor<8xi32>, %arg2: tensor<8xf32>, %arg3: tensor<8xf32>) -> (tensor<?x128xf32>, tensor<?xi32>, tensor<4xi32>, tensor<?xf32>) {
  %activeNum = mrt.constant.i64 4 : !mrt.i64
  %expertCapacity = mrt.constant.i64 32 : !mrt.i64
  %expertNum = mrt.constant.i64 4 : !mrt.i64
  %dropPadMode = mrt.constant.i64 0 : !mrt.i64
  %expertTokensNumType = mrt.constant.i64 0 : !mrt.i64
  %expertTokensNumFlag = mrt.constant.boolean false : !mrt.boolean
  %quantMode = mrt.constant.i64 0 : !mrt.i64
  %activeExpertRangeOptional = mrt.constant.i64_array [0, 1, 2, 3] : !mrt.i64_array
  %rowIdxType = mrt.constant.i64 0 : !mrt.i64
  %0, %1, %2, %3 = mrt.moe_init_routing_v3 %arg0, %arg1, %arg2, %arg3, %activeNum, %expertCapacity, %expertNum, %dropPadMode, %expertTokensNumType, %expertTokensNumFlag, %quantMode, %activeExpertRangeOptional, %rowIdxType : (tensor<8x128xf32>, tensor<8xi32>, tensor<8xf32>, tensor<8xf32>, !mrt.i64, !mrt.i64, !mrt.i64, !mrt.i64, !mrt.i64, !mrt.boolean, !mrt.i64, !mrt.i64_array, !mrt.i64) -> (tensor<?x128xf32>, tensor<?xi32>, tensor<4xi32>, tensor<?xf32>)
  return %0, %1, %2, %3 : tensor<?x128xf32>, tensor<?xi32>, tensor<4xi32>, tensor<?xf32>
}
