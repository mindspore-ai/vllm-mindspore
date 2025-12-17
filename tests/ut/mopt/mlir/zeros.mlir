module {
  func.func @main() -> !torch.vtensor<[10,10],f16> {
    %false = torch.constant.bool false
    %none = torch.constant.none
    %int5 = torch.constant.int 5
    %int10 = torch.constant.int 10
    %0 = torch.prim.ListConstruct %int10, %int10 : (!torch.int, !torch.int) -> !torch.list<int>
    %npu = torch.constant.device "npu"
    %1 = torch.aten.zeros %0, %int5, %none, %npu, %false : !torch.list<int>, !torch.int, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[10,10],f16>
    return %1 : !torch.vtensor<[10,10],f16>
  }
}
