
import pytest
import torch
from mrt.torch.fx_backend import backend, register_dvm_op
from tests.mark_utils import arg_mark


def _get_mrt_dvm_lib():
    # torch.library.Library("mrt_dvm", "DEF") can only be created once per process.
    # If other tests already created it, fall back to "FRAGMENT".
    try:
        return torch.library.Library("mrt_dvm", "DEF")
    except Exception:
        return torch.library.Library("mrt_dvm", "FRAGMENT")


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_dvm_call_add_staticshape_e2e():
    """
    Feature: mrt.dvm_call (static shape)
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
    # Note: namespace must be 'mrt_dvm' to trigger the logic in fx_backend
    lib = _get_mrt_dvm_lib()
    lib.define(f"{op_name}(Tensor x, Tensor y) -> Tensor")

    impl_name = f"mrt_dvm::{op_name}"

    @torch.library.impl(lib, op_name, "CompositeExplicitAutograd")
    def dvm_add_impl(x, y):
        # This implementation is for eager mode / fallback
        return x + y

    @torch.library.register_fake(impl_name)
    def dvm_add_fake(x, y):
        return torch.empty_like(x)

    # 3. Define the function to compile
    def test_func(x, y):
        return torch.ops.mrt_dvm.dvm_add_staticshape_mock(x, y)

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
    Feature: mrt.dvm_call (static shape)
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

    lib = _get_mrt_dvm_lib()
    lib.define(f"{op_name}(Tensor x, Tensor y) -> Tensor")

    impl_name = f"mrt_dvm::{op_name}"

    @torch.library.impl(lib, op_name, "CompositeExplicitAutograd")
    def dvm_mul_add_impl(x, y):
        return x * y + x

    @torch.library.register_fake(impl_name)
    def dvm_mul_add_fake(x, y):
        return torch.empty_like(x)

    def test_func(x, y):
        return torch.ops.mrt_dvm.dvm_mul_add_fused_staticshape_mock(x, y)

    compiled_func = torch.compile(test_func, backend=backend)

    x = torch.randn(4, 8, device="npu", dtype=torch.float16)
    y = torch.randn(4, 8, device="npu", dtype=torch.float16)

    res = compiled_func(x, y)
    expected = x * y + x
    torch.testing.assert_close(res, expected, rtol=1e-2, atol=1e-2)


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("m,k,n", [(1, 32, 256), (16, 128, 64)])
def test_dvm_call_matmul_dynshape_e2e(m, k, n):
    """
    Feature: mrt.dvm_call (dynamic shape)
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

    lib = _get_mrt_dvm_lib()
    lib.define(f"{op_name}(Tensor x, Tensor w) -> Tensor")

    impl_name = f"mrt_dvm::{op_name}"

    @torch.library.impl(lib, op_name, "CompositeExplicitAutograd")
    def dvm_matmul_impl(x, w):
        return x @ w

    @torch.library.register_fake(impl_name)
    def dvm_matmul_fake(x, w):
        # x: (m,k), w: (k,n) -> (m,n)
        return torch.empty((x.shape[0], w.shape[1]), device=x.device, dtype=x.dtype)

    def test_func(x, w):
        return getattr(torch.ops.mrt_dvm, op_name)(x, w)

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
    Feature: mrt.dvm_call (dynamic shape)
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

    lib = _get_mrt_dvm_lib()
    lib.define(f"{op_name}(Tensor x, Tensor w, Tensor y) -> Tensor")

    impl_name = f"mrt_dvm::{op_name}"

    @torch.library.impl(lib, op_name, "CompositeExplicitAutograd")
    def dvm_combo_impl(x, w, y):
        z = x @ w + y
        return torch.exp(z) * z

    @torch.library.register_fake(impl_name)
    def dvm_combo_fake(x, w, y):
        return torch.empty((x.shape[0], w.shape[1]), device=x.device, dtype=x.dtype)

    def test_func(x, w, y):
        return getattr(torch.ops.mrt_dvm, op_name)(x, w, y)

    compiled_func = torch.compile(test_func, backend=backend)

    # Use small magnitude to keep exp stable in float16.
    x = (torch.randn(m, k, device="npu", dtype=torch.float16) * 0.1)
    w = (torch.randn(k, n, device="npu", dtype=torch.float16) * 0.1)
    y = (torch.randn(m, n, device="npu", dtype=torch.float16) * 0.1)

    res = compiled_func(x, w, y)
    z = x @ w + y
    expected = torch.exp(z) * z
    torch.testing.assert_close(res, expected, rtol=1e-2, atol=1e-2)
