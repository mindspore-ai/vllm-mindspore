// RUN: mopt-opt %s | FileCheck %s

func.func @test_unary_ops(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK-LABEL: func.func @test_unary_ops

  // CHECK: dvm.unary Sqrt %{{.*}} : tensor<4xf32> -> tensor<4xf32>
  %0 = dvm.unary Sqrt %arg0 : tensor<4xf32> -> tensor<4xf32>

  // CHECK: dvm.unary Abs %{{.*}} : tensor<4xf32> -> tensor<4xf32>
  %1 = dvm.unary Abs %0 : tensor<4xf32> -> tensor<4xf32>

  // CHECK: dvm.unary Log %{{.*}} : tensor<4xf32> -> tensor<4xf32>
  %2 = dvm.unary Log %1 : tensor<4xf32> -> tensor<4xf32>

  // CHECK: dvm.unary Exp %{{.*}} : tensor<4xf32> -> tensor<4xf32>
  %3 = dvm.unary Exp %2 : tensor<4xf32> -> tensor<4xf32>

  // CHECK: dvm.unary Reciprocal %{{.*}} : tensor<4xf32> -> tensor<4xf32>
  %4 = dvm.unary Reciprocal %3 : tensor<4xf32> -> tensor<4xf32>

  // CHECK: dvm.unary IsFinite %{{.*}} : tensor<4xf32> -> tensor<4xf32>
  %5 = dvm.unary IsFinite %4 : tensor<4xf32> -> tensor<4xf32>

  // CHECK: dvm.unary LogicalNot %{{.*}} : tensor<4xf32> -> tensor<4xf32>
  %6 = dvm.unary LogicalNot %5 : tensor<4xf32> -> tensor<4xf32>

  // CHECK: dvm.unary Round %{{.*}} : tensor<4xf32> -> tensor<4xf32>
  %7 = dvm.unary Round %6 : tensor<4xf32> -> tensor<4xf32>

  // CHECK: dvm.unary Floor %{{.*}} : tensor<4xf32> -> tensor<4xf32>
  %8 = dvm.unary Floor %7 : tensor<4xf32> -> tensor<4xf32>

  // CHECK: dvm.unary Ceil %{{.*}} : tensor<4xf32> -> tensor<4xf32>
  %9 = dvm.unary Ceil %8 : tensor<4xf32> -> tensor<4xf32>

  // CHECK: dvm.unary Trunc %{{.*}} : tensor<4xf32> -> tensor<4xf32>
  %10 = dvm.unary Trunc %9 : tensor<4xf32> -> tensor<4xf32>

  return %10 : tensor<4xf32>
}

func.func @test_binary_ops(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK-LABEL: func.func @test_binary_ops

  // CHECK: dvm.binary Add %{{.*}}, %{{.*}} : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
  %0 = dvm.binary Add %arg0, %arg1 : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>

  // CHECK: dvm.binary Sub %{{.*}}, %{{.*}} : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
  %1 = dvm.binary Sub %0, %arg1 : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>

  // CHECK: dvm.binary Mul %{{.*}}, %{{.*}} : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
  %2 = dvm.binary Mul %1, %arg1 : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>

  // CHECK: dvm.binary Div %{{.*}}, %{{.*}} : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
  %3 = dvm.binary Div %2, %arg1 : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>

  // CHECK: dvm.binary Pow %{{.*}}, %{{.*}} : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
  %4 = dvm.binary Pow %3, %arg1 : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>

  // CHECK: dvm.binary Maximum %{{.*}}, %{{.*}} : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
  %5 = dvm.binary Maximum %4, %arg1 : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>

  // CHECK: dvm.binary Minimum %{{.*}}, %{{.*}} : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
  %6 = dvm.binary Minimum %5, %arg1 : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>

  // CHECK: dvm.binary LogicalAnd %{{.*}}, %{{.*}} : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
  %7 = dvm.binary LogicalAnd %6, %arg1 : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>

  // CHECK: dvm.binary LogicalOr %{{.*}}, %{{.*}} : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
  %8 = dvm.binary LogicalOr %7, %arg1 : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>

  // CHECK: dvm.binary Equal %{{.*}}, %{{.*}} : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
  %9 = dvm.binary Equal %8, %arg1 : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>

  // CHECK: dvm.binary NotEqual %{{.*}}, %{{.*}} : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
  %10 = dvm.binary NotEqual %9, %arg1 : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>

  // CHECK: dvm.binary Greater %{{.*}}, %{{.*}} : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
  %11 = dvm.binary Greater %10, %arg1 : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>

  // CHECK: dvm.binary GreaterEqual %{{.*}}, %{{.*}} : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
  %12 = dvm.binary GreaterEqual %11, %arg1 : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>

  // CHECK: dvm.binary Less %{{.*}}, %{{.*}} : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
  %13 = dvm.binary Less %12, %arg1 : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>

  // CHECK: dvm.binary LessEqual %{{.*}}, %{{.*}} : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
  %14 = dvm.binary LessEqual %13, %arg1 : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>

  return %14 : tensor<4xf32>
}

func.func @test_reduce_ops(%arg0: tensor<4x4xf32>) -> tensor<4xf32> {
  // CHECK-LABEL: func.func @test_reduce_ops

  // CHECK: dvm.reduce Sum %{{.*}} dims [1] keepdims false : tensor<4x4xf32> -> tensor<4xf32>
  %0 = dvm.reduce Sum %arg0 dims [1] keepdims false : tensor<4x4xf32> -> tensor<4xf32>

  // CHECK: dvm.reduce Sum %{{.*}} dims [0, 1] keepdims true : tensor<4x4xf32> -> tensor<1x1xf32>
  %1 = dvm.reduce Sum %arg0 dims [0, 1] keepdims true : tensor<4x4xf32> -> tensor<1x1xf32>

  return %0 : tensor<4xf32>
}

func.func @test_select_op(%cond: tensor<4x1xi1>, %lhs: tensor<4x4xf32>, %rhs: tensor<4x4xf32>) -> tensor<4x4xf32> {
  // CHECK-LABEL: func.func @test_select_op

  // CHECK: dvm.select %{{.*}}, %{{.*}}, %{{.*}} : (tensor<4x1xi1>, tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  %0 = dvm.select %cond, %lhs, %rhs : (tensor<4x1xi1>, tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

func.func @test_cast_op(%arg0: tensor<4x4xf32>) -> tensor<4x4xi32> {
  // CHECK-LABEL: func.func @test_cast_op

  // CHECK: dvm.cast %{{.*}} type Int32 : tensor<4x4xf32> -> tensor<4x4xi32>
  %0 = dvm.cast %arg0 type Int32 : tensor<4x4xf32> -> tensor<4x4xi32>
  return %0 : tensor<4x4xi32>
}

func.func @test_reshape_op(%arg0: tensor<4x4xf32>) -> tensor<16xf32> {
  // CHECK-LABEL: func.func @test_reshape_op

  // CHECK: dvm.reshape %{{.*}} shape [16] : tensor<4x4xf32> -> tensor<16xf32>
  %0 = dvm.reshape %arg0 shape [16] : tensor<4x4xf32> -> tensor<16xf32>
  return %0 : tensor<16xf32>
}

func.func @test_broadcast_op(%arg0: tensor<4x1xf32>) -> tensor<4x4xf32> {
  // CHECK-LABEL: func.func @test_broadcast_op

  // CHECK: dvm.broadcast %{{.*}} shape [4, 4] : tensor<4x1xf32> -> tensor<4x4xf32>
  %0 = dvm.broadcast %arg0 shape [4, 4] : tensor<4x1xf32> -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

func.func @test_matmul_op(%lhs: tensor<4x4xf32>, %rhs: tensor<4x4xf32>, %bias: tensor<4x4xf32>) -> tensor<4x4xf32> {
  // CHECK-LABEL: func.func @test_matmul_op

  // CHECK: dvm.matmul %{{.*}}, %{{.*}} trans_a false trans_b false : tensor<4x4xf32>, tensor<4x4xf32> -> tensor<4x4xf32>
  %0 = dvm.matmul %lhs, %rhs trans_a false trans_b false : tensor<4x4xf32>, tensor<4x4xf32> -> tensor<4x4xf32>

  // CHECK: dvm.matmul %{{.*}}, %{{.*}} trans_a true trans_b true bias %{{.*}} : tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32> -> tensor<4x4xf32>
  %1 = dvm.matmul %lhs, %rhs trans_a true trans_b true bias %bias : tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32> -> tensor<4x4xf32>

  return %1 : tensor<4x4xf32>
}

func.func @test_grouped_matmul_op(%lhs: tensor<4x4xf32>, %rhs: tensor<4x4xf32>, %bias: tensor<4x4xf32>, %group_list: tensor<4xi64>) -> tensor<4x4xf32> {
  // CHECK-LABEL: func.func @test_grouped_matmul_op

  // CHECK: dvm.grouped_matmul %{{.*}}, %{{.*}} trans_a false trans_b false group_list %{{.*}} group_type Split_M : tensor<4x4xf32>, tensor<4x4xf32>, tensor<4xi64> -> tensor<4x4xf32>
  %0 = dvm.grouped_matmul %lhs, %rhs trans_a false trans_b false group_list %group_list group_type Split_M : tensor<4x4xf32>, tensor<4x4xf32>, tensor<4xi64> -> tensor<4x4xf32>

  // CHECK: dvm.grouped_matmul %{{.*}}, %{{.*}} trans_a true trans_b true group_list %{{.*}} group_type Split_K bias %{{.*}} : tensor<4x4xf32>, tensor<4x4xf32>, tensor<4xi64>, tensor<4x4xf32> -> tensor<4x4xf32>
  %1 = dvm.grouped_matmul %lhs, %rhs trans_a true trans_b true group_list %group_list group_type Split_K bias %bias : tensor<4x4xf32>, tensor<4x4xf32>, tensor<4xi64>, tensor<4x4xf32> -> tensor<4x4xf32>

  return %1 : tensor<4x4xf32>
}

func.func @test_copy_op(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
  // CHECK-LABEL: func.func @test_copy_op

  // CHECK: dvm.copy %{{.*}} : tensor<4x4xf32> -> tensor<4x4xf32>
  %0 = dvm.copy %arg0 : tensor<4x4xf32> -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

func.func @test_onehot_op(%indices: tensor<4xi32>) -> tensor<4x10xf32> {
  // CHECK-LABEL: func.func @test_onehot_op

  // CHECK: dvm.onehot %{{.*}} depth [10] axis 1 {off_value = 0.000000e+00 : f64, on_value = 1.000000e+00 : f64} : tensor<4xi32> -> tensor<4x10xf32>
  %0 = dvm.onehot %indices depth [10] axis 1 {on_value = 1.0 : f64, off_value = 0.0 : f64} : tensor<4xi32> -> tensor<4x10xf32>
  return %0 : tensor<4x10xf32>
}

func.func @test_elemany_op(%arg0: tensor<4x4xi1>) -> tensor<i1> {
  // CHECK-LABEL: func.func @test_elemany_op

  // CHECK: dvm.elemany %{{.*}} : tensor<4x4xi1> -> tensor<i1>
  %0 = dvm.elemany %arg0 : tensor<4x4xi1> -> tensor<i1>
  return %0 : tensor<i1>
}

func.func @test_load_store_ops(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
  // CHECK-LABEL: func.func @test_load_store_ops

  // CHECK: dvm.store %{{.*}} : tensor<4x4xf32> -> tensor<4x4xf32>
  %0 = dvm.store %arg0 : tensor<4x4xf32> -> tensor<4x4xf32>

  // CHECK: dvm.load %{{.*}} : tensor<4x4xf32> -> tensor<4x4xf32>
  %1 = dvm.load %0 : tensor<4x4xf32> -> tensor<4x4xf32>

  return %1 : tensor<4x4xf32>
}