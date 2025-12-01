// RUN: mopt-opt --annotate-linalg-for-hacc %s | FileCheck %s

// Test 1: Function with Linalg operations should be annotated
// CHECK-LABEL: func.func @test_with_linalg
// CHECK-SAME: attributes {hacc.entry, hacc.function_kind = "#hacc.function_kind<HOST>"}
func.func @test_with_linalg(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %0 = tensor.empty() : tensor<4x8xf32>
  %cst = arith.constant 0.0 : f32
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<4x8xf32>) -> tensor<4x8xf32>
  return %1 : tensor<4x8xf32>
}

// Test 2: Function with linalg.generic should be annotated
// CHECK-LABEL: func.func @test_with_generic
// CHECK-SAME: attributes {hacc.entry, hacc.function_kind = "#hacc.function_kind<HOST>"}
func.func @test_with_generic(%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %0 = tensor.empty() : tensor<4x8xf32>
  %1 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%arg0, %arg1 : tensor<4x8xf32>, tensor<4x8xf32>)
    outs(%0 : tensor<4x8xf32>) {
  ^bb0(%in0: f32, %in1: f32, %out: f32):
    %2 = arith.addf %in0, %in1 : f32
    linalg.yield %2 : f32
  } -> tensor<4x8xf32>
  return %1 : tensor<4x8xf32>
}

// Test 3: Function without Linalg operations should NOT be annotated
// CHECK-LABEL: func.func @test_without_linalg
// CHECK-NOT: hacc.entry
// CHECK-NOT: hacc.function_kind
func.func @test_without_linalg(%arg0: f32, %arg1: f32) -> f32 {
  %0 = arith.addf %arg0, %arg1 : f32
  return %0 : f32
}

// Test 4: Already annotated function should remain unchanged (idempotency test)
// CHECK-LABEL: func.func @test_already_annotated
// CHECK-SAME: attributes {hacc.entry, hacc.function_kind = "#hacc.function_kind<HOST>"}
// CHECK-NOT: hacc.entry, hacc.entry
func.func @test_already_annotated(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32>
    attributes {hacc.entry, hacc.function_kind = "#hacc.function_kind<HOST>"} {
  %0 = tensor.empty() : tensor<4x8xf32>
  %cst = arith.constant 0.0 : f32
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<4x8xf32>) -> tensor<4x8xf32>
  return %1 : tensor<4x8xf32>
}

// Test 5: Function with matmul (another Linalg op) should be annotated
// CHECK-LABEL: func.func @test_with_matmul
// CHECK-SAME: attributes {hacc.entry, hacc.function_kind = "#hacc.function_kind<HOST>"}
func.func @test_with_matmul(%arg0: tensor<4x8xf32>, %arg1: tensor<8x16xf32>) -> tensor<4x16xf32> {
  %0 = tensor.empty() : tensor<4x16xf32>
  %1 = linalg.matmul ins(%arg0, %arg1 : tensor<4x8xf32>, tensor<8x16xf32>)
                     outs(%0 : tensor<4x16xf32>) -> tensor<4x16xf32>
  return %1 : tensor<4x16xf32>
}

// Test 6: Nested function calls - only annotate functions with Linalg ops
// CHECK-LABEL: func.func @helper_with_arith
// CHECK-NOT: hacc.entry
func.func @helper_with_arith(%arg0: f32) -> f32 {
  %0 = arith.mulf %arg0, %arg0 : f32
  return %0 : f32
}

// CHECK-LABEL: func.func @main_with_linalg
// CHECK-SAME: attributes {hacc.entry, hacc.function_kind = "#hacc.function_kind<HOST>"}
func.func @main_with_linalg(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %0 = tensor.empty() : tensor<4x8xf32>
  %cst = arith.constant 1.0 : f32
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<4x8xf32>) -> tensor<4x8xf32>
  return %1 : tensor<4x8xf32>
}

