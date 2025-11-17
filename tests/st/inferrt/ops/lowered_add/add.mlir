// Add MLIR
// Element-wise addition of two tensors

module {
    func.func @add_kernel(
        %input: tensor<1x6144xf16>,
        %bias: tensor<1x6144xf16>,
        %output: tensor<1x6144xf16>) -> tensor<1x6144xf16>
            attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<HOST>}
    {
        %result = linalg.elemwise_binary {func = #linalg.binary_fn<add>} ins(%input, %bias : tensor<1x6144xf16>, tensor<1x6144xf16>) outs(%output : tensor<1x6144xf16>) -> tensor<1x6144xf16>
        return %result : tensor<1x6144xf16>
    }
}
