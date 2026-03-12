"""Test DVM call end-to-end operations."""

import pytest
import torch
from ms_inferrt.torch.fx_backend import backend, register_dvm_op
from tests.mark_utils import arg_mark


def _get_ms_inferrt_dvm_lib():
    # torch.library.Library("ms_inferrt_dvm", "DEF") can only be created once per process.
    # If other tests already created it, fall back to "FRAGMENT".
    try:
        return torch.library.Library("ms_inferrt_dvm", "DEF")
    except Exception:
        return torch.library.Library("ms_inferrt_dvm", "FRAGMENT")


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_dvm_call_add_staticshape_e2e():
    """
    Feature: ms_inferrt.dvm_call (static shape)
    Description: Single-op graph: out = x + y
    Expectation: DVM kernel executes and matches eager result.
    """
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU not available")

    # 1. Register a DVM payload matching the OpCode format
    # This payload implements: out = x + y
    op_name = "dvm_add_staticshape_mock"
    payload_json = '''{
        "version": 1,
        "kernel_type": "static_shape",
        "instructions": [
            {"op": "load", "idx": 0, "inputs": [], "attrs": {"dtype": "f16"}},
            {"op": "load", "idx": 1, "inputs": [], "attrs": {"dtype": "f16"}},
            {"op": "binary", "idx": 2, "inputs": [0, 1], "attrs": {"type": "Add"}},
            {"op": "store", "idx": 3, "inputs": [2], "attrs": {}}
        ],
        "input_indices": [0, 1],
        "output_indices": [3]
    }'''
    register_dvm_op(op_name, payload_json)

    # 2. Define the custom op in torch
    # Note: namespace must be 'ms_inferrt_dvm' to trigger the logic in fx_backend
    lib = _get_ms_inferrt_dvm_lib()
    lib.define(f"{op_name}(Tensor x, Tensor y) -> Tensor")

    impl_name = f"ms_inferrt_dvm::{op_name}"

    @torch.library.impl(lib, op_name, "CompositeExplicitAutograd")
    def dvm_add_impl(x, y):
        # This implementation is for eager mode / fallback
        return x + y

    @torch.library.register_fake(impl_name)
    def dvm_add_fake(x, _y):
        return torch.empty_like(x)

    # 3. Define the function to compile
    def test_func(x, y):
        return torch.ops.ms_inferrt_dvm.dvm_add_staticshape_mock(x, y)

    # 4. Compile with mrt backend
    compiled_func = torch.compile(test_func, backend=backend)

    # 5. Run
    x = torch.randn(4, 4, device="npu", dtype=torch.float16)
    y = torch.randn(4, 4, device="npu", dtype=torch.float16)

    # This invokes GraphExecutor -> OpDvmCall -> DvmOp.
    res = compiled_func(x, y)

    # Verify result
    expected = x + y
    torch.testing.assert_close(res, expected, rtol=1e-2, atol=1e-2)


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_dvm_call_mul_add_fused_staticshape_e2e():
    """
    Feature: ms_inferrt.dvm_call (static shape)
    Description: Fused graph: out = x * y + x
    Expectation: DVM kernel executes fused mul+add and matches eager result.
    """
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU not available")

    op_name = "dvm_mul_add_fused_staticshape_mock"
    payload_json = '''{
        "version": 1,
        "kernel_type": "static_shape",
        "instructions": [
            {"op": "load", "idx": 0, "inputs": [], "attrs": {"dtype": "f16"}},
            {"op": "load", "idx": 1, "inputs": [], "attrs": {"dtype": "f16"}},
            {"op": "binary", "idx": 2, "inputs": [0, 1], "attrs": {"type": "Mul"}},
            {"op": "binary", "idx": 3, "inputs": [2, 0], "attrs": {"type": "Add"}},
            {"op": "store", "idx": 4, "inputs": [3], "attrs": {}}
        ],
        "input_indices": [0, 1],
        "output_indices": [4]
    }'''
    register_dvm_op(op_name, payload_json)

    lib = _get_ms_inferrt_dvm_lib()
    lib.define(f"{op_name}(Tensor x, Tensor y) -> Tensor")

    impl_name = f"ms_inferrt_dvm::{op_name}"

    @torch.library.impl(lib, op_name, "CompositeExplicitAutograd")
    def dvm_mul_add_impl(x, y):
        return x * y + x

    @torch.library.register_fake(impl_name)
    def dvm_mul_add_fake(x, _y):
        return torch.empty_like(x)

    def test_func(x, y):
        return torch.ops.ms_inferrt_dvm.dvm_mul_add_fused_staticshape_mock(x, y)

    compiled_func = torch.compile(test_func, backend=backend)

    x = torch.randn(4, 8, device="npu", dtype=torch.float16)
    y = torch.randn(4, 8, device="npu", dtype=torch.float16)

    res = compiled_func(x, y)
    expected = x * y + x
    torch.testing.assert_close(res, expected, rtol=1e-2, atol=1e-2)


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_dvm_call_two_outputs_staticshape_e2e():
    """
    Feature: ms_inferrt.dvm_call (static shape, multi-output)
    Description: Fused graph with two outputs:
      out0 = x + y
      out1 = x * y
    Expectation: DVM kernel executes and returns a tuple(Tensor, Tensor) matching eager results.
    """
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU not available")

    op_name = "dvm_two_outputs_staticshape_mock"
    payload_json = '''{
        "version": 1,
        "kernel_type": "static_shape",
        "instructions": [
            {"op": "load", "idx": 0, "inputs": [], "attrs": {"dtype": "f16"}},
            {"op": "load", "idx": 1, "inputs": [], "attrs": {"dtype": "f16"}},
            {"op": "binary", "idx": 2, "inputs": [0, 1], "attrs": {"type": "Add"}},
            {"op": "binary", "idx": 3, "inputs": [0, 1], "attrs": {"type": "Mul"}},
            {"op": "store", "idx": 4, "inputs": [2], "attrs": {}},
            {"op": "store", "idx": 5, "inputs": [3], "attrs": {}}
        ],
        "input_indices": [0, 1],
        "output_indices": [4, 5]
    }'''
    register_dvm_op(op_name, payload_json)

    lib = _get_ms_inferrt_dvm_lib()
    lib.define(f"{op_name}(Tensor x, Tensor y) -> (Tensor, Tensor)")

    impl_name = f"ms_inferrt_dvm::{op_name}"

    @torch.library.impl(lib, op_name, "CompositeExplicitAutograd")
    def dvm_two_outputs_impl(x, y):
        return (x + y, x * y)

    @torch.library.register_fake(impl_name)
    def dvm_two_outputs_fake(x, _y):
        return (torch.empty_like(x), torch.empty_like(x))

    def test_func(x, y):
        return torch.ops.ms_inferrt_dvm.dvm_two_outputs_staticshape_mock(x, y)

    compiled_func = torch.compile(test_func, backend=backend)

    x = torch.randn(4, 4, device="npu", dtype=torch.float16)
    y = torch.randn(4, 4, device="npu", dtype=torch.float16)

    out0, out1 = compiled_func(x, y)
    torch.testing.assert_close(out0, x + y, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(out1, x * y, rtol=1e-2, atol=1e-2)


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_dvm_call_two_outputs_output_indices_order_guard_e2e():
    """
    Feature: ms_inferrt.dvm_call multi-output output ordering.
    Description: Construct a payload where store(Add) appears before store(Mul) in the instruction stream,
    but output_indices order is [store(Mul), store(Add)]. This guards against mapping outputs by store
    appearance order instead of payload.output_indices.
    Expectation: Output tuple order follows payload.output_indices (out0==x*y, out1==x+y); no swap.
    """
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU not available")

    op_name = "dvm_two_outputs_output_indices_order_guard_staticshape_mock"
    payload_json = '''{
        "version": 1,
        "kernel_type": "static_shape",
        "instructions": [
            {"op": "load", "idx": 0, "inputs": [], "attrs": {"dtype": "f16"}},
            {"op": "load", "idx": 1, "inputs": [], "attrs": {"dtype": "f16"}},
            {"op": "binary", "idx": 2, "inputs": [0, 1], "attrs": {"type": "Add"}},
            {"op": "binary", "idx": 3, "inputs": [0, 1], "attrs": {"type": "Mul"}},
            {"op": "store", "idx": 4, "inputs": [2], "attrs": {}},
            {"op": "store", "idx": 5, "inputs": [3], "attrs": {}}
        ],
        "input_indices": [0, 1],
        "output_indices": [5, 4]
    }'''
    register_dvm_op(op_name, payload_json)

    lib = _get_ms_inferrt_dvm_lib()
    lib.define(f"{op_name}(Tensor x, Tensor y) -> (Tensor, Tensor)")

    impl_name = f"ms_inferrt_dvm::{op_name}"

    @torch.library.impl(lib, op_name, "CompositeExplicitAutograd")
    def dvm_two_outputs_impl(x, y):
        # Must match payload.output_indices order: [Mul, Add]
        return (x * y, x + y)

    @torch.library.register_fake(impl_name)
    def dvm_two_outputs_fake(x, _y):
        return (torch.empty_like(x), torch.empty_like(x))

    def test_func(x, y):
        return getattr(torch.ops.ms_inferrt_dvm, op_name)(x, y)

    compiled_func = torch.compile(test_func, backend=backend)

    x = torch.randn(4, 4, device="npu", dtype=torch.float16)
    y = torch.randn(4, 4, device="npu", dtype=torch.float16)

    out0, out1 = compiled_func(x, y)
    torch.testing.assert_close(out0, x * y, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(out1, x + y, rtol=1e-2, atol=1e-2)


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_dvm_call_two_outputs_shape_ref_out_pos_e2e():
    """
    Feature: ms_inferrt.dvm_call multi-output output-shape ref selection for reshape/broadcast.
    Description:
      In multi-output graphs, reshape/broadcast need a *target output shape* (ShapeRef).
      output_indices only defines which stored values map to outputs; it does NOT tell which output
      shape a given intermediate op should use (because store produces a new value idx).
      We extend schema with attrs.shape_ref (mutually exclusive fields) to explicitly bind
      reshape/broadcast target shape source.
    Expectation:
      out0 == x.reshape(-1) (shape [m*k]), out1 == x (shape [m,k]).
    """
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU not available")

    op_name = "dvm_two_outputs_shape_ref_out_pos_staticshape_mock"
    payload_json = '''{
        "version": 1,
        "kernel_type": "static_shape",
        "instructions": [
            {"op": "load", "idx": 0, "inputs": [], "attrs": {"dtype": "f16"}},
            {"op": "reshape", "idx": 1, "inputs": [0], "attrs": {"shape_ref": {"output_pos": 0}}},
            {"op": "store", "idx": 2, "inputs": [1], "attrs": {}},

            {"op": "load", "idx": 3, "inputs": [], "attrs": {"dtype": "f16"}},
            {"op": "broadcast", "idx": 4, "inputs": [0], "attrs": {"shape_ref": {"output_pos": 1}}},
            {"op": "store", "idx": 5, "inputs": [4], "attrs": {}}
        ],
        "input_indices": [0, 3],
        "output_indices": [2, 5]
    }'''
    register_dvm_op(op_name, payload_json)

    lib = _get_ms_inferrt_dvm_lib()
    lib.define(f"{op_name}(Tensor x, Tensor y) -> (Tensor, Tensor)")
    impl_name = f"ms_inferrt_dvm::{op_name}"

    @torch.library.impl(lib, op_name, "CompositeExplicitAutograd")
    def impl(x, y):
        # out0: flatten; out1: identity
        return (x.reshape(-1), x)

    @torch.library.register_fake(impl_name)
    def fake(x, _y):
        # Shapes must match the intended outputs to provide correct ShapeRefs to DVM build.
        return (torch.empty((x.numel(),), device=x.device, dtype=x.dtype), torch.empty_like(x))

    def test_func(x, y):
        return getattr(torch.ops.ms_inferrt_dvm, op_name)(x, y)

    compiled_func = torch.compile(test_func, backend=backend)

    x = torch.randn(2, 3, device="npu", dtype=torch.float16)
    y = torch.randn(2, 3, device="npu", dtype=torch.float16)
    out0, out1 = compiled_func(x, y)
    torch.testing.assert_close(out0, x.reshape(-1), rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(out1, x, rtol=1e-2, atol=1e-2)


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_dvm_call_shape_ref_const_dims_e2e():
    """
    Feature: ms_inferrt.dvm_call shape_ref const dims.
    Description:
      reshape target shape is a constant carried by payload (attrs.shape_ref.dims), i.e. not an output shape.
    Expectation:
      out == x.reshape(2, 3) for a 2x3 input.
    """
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU not available")

    op_name = "dvm_shape_ref_const_dims_staticshape_mock"
    payload_json = '''{
        "version": 1,
        "kernel_type": "static_shape",
        "instructions": [
            {"op": "load", "idx": 0, "inputs": [], "attrs": {"dtype": "f16"}},
            {"op": "reshape", "idx": 1, "inputs": [0], "attrs": {"shape_ref": {"dims": [2, 3]}}},
            {"op": "store", "idx": 2, "inputs": [1], "attrs": {}}
        ],
        "input_indices": [0],
        "output_indices": [2]
    }'''
    register_dvm_op(op_name, payload_json)

    lib = _get_ms_inferrt_dvm_lib()
    lib.define(f"{op_name}(Tensor x) -> Tensor")
    impl_name = f"ms_inferrt_dvm::{op_name}"

    @torch.library.impl(lib, op_name, "CompositeExplicitAutograd")
    def impl(x):
        return x.reshape(2, 3)

    @torch.library.register_fake(impl_name)
    def fake(x):
        return torch.empty((2, 3), device=x.device, dtype=x.dtype)

    def test_func(x):
        return getattr(torch.ops.ms_inferrt_dvm, op_name)(x)

    compiled_func = torch.compile(test_func, backend=backend)

    x = torch.randn(2, 3, device="npu", dtype=torch.float16)
    out = compiled_func(x)
    torch.testing.assert_close(out, x.reshape(2, 3), rtol=1e-2, atol=1e-2)


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("m,k,n", [(1, 32, 256), (16, 128, 64)])
def test_dvm_call_matmul_dynshape_e2e(m, k, n):
    """
    Feature: ms_inferrt.dvm_call (dynamic shape)
    Description: Single-op graph: out = x @ w
    Expectation: DVM kernel executes MatMul and matches eager result for varying shapes.
    """
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU not available")

    # Parametrized test: keep op_name unique per invocation to avoid re-defining the same schema.
    op_name = f"dvm_matmul_dynshape_mock_{m}_{k}_{n}"
    payload_json = '''{
        "version": 1,
        "kernel_type": "dyn_shape",
        "instructions": [
            {"op": "load", "idx": 0, "inputs": [], "attrs": {"dtype": "f16"}},
            {"op": "load", "idx": 1, "inputs": [], "attrs": {"dtype": "f16"}},
            {"op": "matmul", "idx": 2, "inputs": [0, 1], "attrs": {"trans_a": false, "trans_b": false}},
            {"op": "store", "idx": 3, "inputs": [2], "attrs": {}}
        ],
        "input_indices": [0, 1],
        "output_indices": [3]
    }'''
    register_dvm_op(op_name, payload_json)

    lib = _get_ms_inferrt_dvm_lib()
    lib.define(f"{op_name}(Tensor x, Tensor w) -> Tensor")

    impl_name = f"ms_inferrt_dvm::{op_name}"

    @torch.library.impl(lib, op_name, "CompositeExplicitAutograd")
    def dvm_matmul_impl(x, w):
        return x @ w

    @torch.library.register_fake(impl_name)
    def dvm_matmul_fake(x, w):
        # x: (m,k), w: (k,n) -> (m,n)
        return torch.empty((x.shape[0], w.shape[1]), device=x.device, dtype=x.dtype)

    def test_func(x, w):
        return getattr(torch.ops.ms_inferrt_dvm, op_name)(x, w)

    compiled_func = torch.compile(test_func, backend=backend)

    x = torch.randn(m, k, device="npu", dtype=torch.float16)
    w = torch.randn(k, n, device="npu", dtype=torch.float16)

    res = compiled_func(x, w)
    expected = x @ w
    torch.testing.assert_close(res, expected, rtol=1e-2, atol=1e-2)


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("m,k,n", [(1, 32, 256), (16, 128, 64)])
def test_dvm_call_dynshape_combo_with_matmul_e2e(m, k, n):
    """
    Feature: ms_inferrt.dvm_call (dynamic shape)
    Description: Combo graph with MatMul + elementwise fusion:
      z = x @ w + y
      out = exp(z) * z
    Expectation: Correct result for varying (m,k,n) shapes.
    """
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU not available")

    # Parametrized test: keep op_name unique per invocation to avoid re-defining the same schema.
    op_name = f"dvm_combo_dynshape_with_matmul_mock_{m}_{k}_{n}"
    payload_json = '''{
        "version": 1,
        "kernel_type": "dyn_shape",
        "instructions": [
            {"op": "load", "idx": 0, "inputs": [], "attrs": {"dtype": "f16"}},
            {"op": "load", "idx": 1, "inputs": [], "attrs": {"dtype": "f16"}},
            {"op": "load", "idx": 2, "inputs": [], "attrs": {"dtype": "f16"}},
            {"op": "matmul", "idx": 3, "inputs": [0, 1], "attrs": {"trans_a": false, "trans_b": false}},
            {"op": "binary", "idx": 4, "inputs": [3, 2], "attrs": {"type": "Add"}},
            {"op": "unary", "idx": 5, "inputs": [4], "attrs": {"type": "Exp"}},
            {"op": "binary", "idx": 6, "inputs": [5, 4], "attrs": {"type": "Mul"}},
            {"op": "store", "idx": 7, "inputs": [6], "attrs": {}}
        ],
        "input_indices": [0, 1, 2],
        "output_indices": [7]
    }'''
    register_dvm_op(op_name, payload_json)

    lib = _get_ms_inferrt_dvm_lib()
    lib.define(f"{op_name}(Tensor x, Tensor w, Tensor y) -> Tensor")

    impl_name = f"ms_inferrt_dvm::{op_name}"

    @torch.library.impl(lib, op_name, "CompositeExplicitAutograd")
    def dvm_combo_impl(x, w, y):
        z = x @ w + y
        return torch.exp(z) * z

    @torch.library.register_fake(impl_name)
    def dvm_combo_fake(x, w, _y):
        return torch.empty((x.shape[0], w.shape[1]), device=x.device, dtype=x.dtype)

    def test_func(x, w, y):
        return getattr(torch.ops.ms_inferrt_dvm, op_name)(x, w, y)

    compiled_func = torch.compile(test_func, backend=backend)

    # Use small magnitude to keep exp stable in float16.
    x = torch.randn(m, k, device="npu", dtype=torch.float16) * 0.1
    w = torch.randn(k, n, device="npu", dtype=torch.float16) * 0.1
    y = torch.randn(m, n, device="npu", dtype=torch.float16) * 0.1

    res = compiled_func(x, w, y)
    z = x @ w + y
    expected = torch.exp(z) * z
    torch.testing.assert_close(res, expected, rtol=1e-2, atol=1e-2)
