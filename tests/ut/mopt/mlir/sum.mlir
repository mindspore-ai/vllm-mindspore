// default sum
module {
  func.func @main(%arg0: !torch.vtensor<[2,3,4],f16>) -> !torch.vtensor<[1,3,4],f16> {
    %none = torch.constant.none
    %true = torch.constant.bool true
    %int0 = torch.constant.int 0
    %0 = torch.prim.ListConstruct %int0 : (!torch.int) -> !torch.list<int>
    %1 = torch.aten.sum.dim_IntList %arg0, %0, %true, %none : !torch.vtensor<[2,3,4],f16>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[1,3,4],f16>
    return %1 : !torch.vtensor<[1,3,4],f16>
  }
}

// sum with dtype, dim is empty tuple
module {
  func.func @main(%arg0: !torch.vtensor<[8,32],f16>) -> !torch.vtensor<[],f32> {
    %int6 = torch.constant.int 6
    %false = torch.constant.bool false
    %0 = torch.prim.ListConstruct  : () -> !torch.list<int>
    %1 = torch.aten.sum.dim_IntList %arg0, %0, %false, %int6 : !torch.vtensor<[8,32],f16>, !torch.list<int>, !torch.bool, !torch.int -> !torch.vtensor<[],f32>
    return %1 : !torch.vtensor<[],f32>
  }
}

// dynamic shape
module {
  func.func @main(%arg0: !torch.int, %arg1: !torch.int, %arg2: !torch.int, %arg3: !torch.vtensor<[?,?,?],f16>) -> !torch.vtensor<[1,?,?],f16> {
    %none = torch.constant.none
    %true = torch.constant.bool true
    %int0 = torch.constant.int 0
    %0 = torch.symbolic_int "s0" {min_val = 2, max_val = 9223372036854775807} : !torch.int
    %1 = torch.symbolic_int "s1" {min_val = 2, max_val = 9223372036854775807} : !torch.int
    %2 = torch.symbolic_int "s2" {min_val = 2, max_val = 9223372036854775807} : !torch.int
    torch.bind_symbolic_shape %arg3, [%0, %1, %2], affine_map<()[s0, s1, s2] -> (s0, s1, s2)> : !torch.vtensor<[?,?,?],f16>
    %3 = torch.prim.ListConstruct %int0 : (!torch.int) -> !torch.list<int>
    %4 = torch.aten.sum.dim_IntList %arg3, %3, %true, %none : !torch.vtensor<[?,?,?],f16>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[1,?,?],f16>
    torch.bind_symbolic_shape %4, [%1, %2], affine_map<()[s0, s1] -> (1, s0, s1)> : !torch.vtensor<[1,?,?],f16>
    return %4 : !torch.vtensor<[1,?,?],f16>
  }
}

// dim is none
module {
  func.func @main(%arg0: !torch.vtensor<[2,3,4],f16>) -> !torch.vtensor<[1,1,1],f16> {
    %none = torch.constant.none
    %true = torch.constant.bool true
    %0 = torch.aten.sum.dim_IntList %arg0, %none, %true, %none : !torch.vtensor<[2,3,4],f16>, !torch.none, !torch.bool, !torch.none -> !torch.vtensor<[1,1,1],f16>
    return %0 : !torch.vtensor<[1,1,1],f16>
  }
}
