module {
  func.func @main(%arg0: !torch.vtensor<[2,5],f32>, %arg1: !torch.vtensor<[2,5],f32>) -> !torch.vtensor<[4,5],f32> {
    %int0 = torch.constant.int 0
    %0 = torch.prim.ListConstruct %arg1, %arg0 : (!torch.vtensor<[2,5],f32>, !torch.vtensor<[2,5],f32>) -> !torch.list<vtensor>
    %1 = torch.aten.cat %0, %int0 : !torch.list<vtensor>, !torch.int -> !torch.vtensor<[4,5],f32>
    return %1 : !torch.vtensor<[4,5],f32>
  }
}
