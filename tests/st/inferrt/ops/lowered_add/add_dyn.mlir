// Add MLIR (Dynamic Shape)
// Element-wise addition of two tensors

module {
    func.func @add_dyn_kernel(
        %input: tensor<?x?xf16>,
        %bias: tensor<?x?xf16>,
        %output: tensor<?x?xf16>) -> tensor<?x?xf16>
            attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<HOST>}
    {
        %result = linalg.elemwise_binary {func = #linalg.binary_fn<add>} ins(%input, %bias : tensor<?x?xf16>, tensor<?x?xf16>) outs(%output : tensor<?x?xf16>) -> tensor<?x?xf16>
        return %result : tensor<?x?xf16>
    }
}
