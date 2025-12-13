module {
  func.func @main() -> !torch.vtensor<[10,10],f16> {
    %false = torch.constant.bool false
    %none = torch.constant.none
    %int5 = torch.constant.int 5
    %int10 = torch.constant.int 10
    %0 = torch.prim.ListConstruct %int10, %int10 : (!torch.int, !torch.int) -> !torch.list<int>
    %cpu = torch.constant.device "cpu"
    %1 = torch.aten.empty.memory_format %0, %int5, %none, %cpu, %false, %none : !torch.list<int>, !torch.int, !torch.none, !torch.Device, !torch.bool, !torch.none -> !torch.vtensor<[10,10],f16>
    return %1 : !torch.vtensor<[10,10],f16>
  }
}

module {
  func.func @main(%arg0: !torch.int, %arg1: !torch.int) -> !torch.vtensor<[?,?],f16> {
    %false = torch.constant.bool false
    %none = torch.constant.none
    %int5 = torch.constant.int 5
    %0 = torch.symbolic_int "s0" {min_val = 0, max_val = 9223372036854775807} : !torch.int
    %1 = torch.symbolic_int "s1" {min_val = 1, max_val = 9223372036854775807} : !torch.int
    %2 = torch.prim.ListConstruct %arg0, %arg1 : (!torch.int, !torch.int) -> !torch.list<int>
    %cpu = torch.constant.device "cpu"
    %3 = torch.aten.empty.memory_format %2, %int5, %none, %cpu, %false, %none : !torch.list<int>, !torch.int, !torch.none, !torch.Device, !torch.bool, !torch.none -> !torch.vtensor<[?,?],f16>
    torch.bind_symbolic_shape %3, [%0, %1], affine_map<()[s0, s1] -> (s0, s1)> : !torch.vtensor<[?,?],f16>
    return %3 : !torch.vtensor<[?,?],f16>
  }
}

module {
  func.func @main() -> !torch.vtensor<[10,10],f16> {
    %false = torch.constant.bool false
    %none = torch.constant.none
    %int5 = torch.constant.int 5
    %int10 = torch.constant.int 10
    %0 = torch.prim.ListConstruct %int10, %int10 : (!torch.int, !torch.int) -> !torch.list<int>
    %npu = torch.constant.device "npu:0"
    %1 = torch.aten.empty.memory_format %0, %int5, %none, %npu, %false, %none : !torch.list<int>, !torch.int, !torch.none, !torch.Device, !torch.bool, !torch.none -> !torch.vtensor<[10,10],f16>
    return %1 : !torch.vtensor<[10,10],f16>
  }
}
