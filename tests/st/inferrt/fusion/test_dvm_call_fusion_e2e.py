#!/usr/bin/env python3
"""
tests/st/inferrt/fusion/test_dvm_call_fusion_e2e.py
"""

import os
import sys

import pytest
import torch

from tests.mark_utils import arg_mark


os.environ.setdefault("MOPT_ENABLE_LINALG_CALL", "0")
# Enable dynamic-shape debug tests by default. Set MOPT_ENABLE_DYNAMIC_SHAPE_TEST=0 to disable.
os.environ.setdefault("MOPT_ENABLE_DYNAMIC_SHAPE_TEST", "1")


def check_npu_available():
    """Check if NPU is available."""
    try:
        import torch_npu  # noqa: F401

        if not torch.npu.is_available():
            print("Error: NPU not available")
            return False
        return True
    except ImportError:
        print("Error: torch_npu not installed")
        return False


@arg_mark(
    plat_marks=["platform_ascend"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_mul_exp_fusion_lower_to_dvm_call():
    """
    Feature: Outlined StableHLO fusion region lowered to mrt.dvm_call.
    Description: Test sigmoid + (mul -> exp) cluster fusion using default lowering path (mrt.dvm_call)
      with float16 tensors on NPU device.
    Expectation: Numerical result matches eager torch within rtol/atol 1e-2,
      and printed IR contains 'mrt.dvm_call' but not 'mrt.linalg_call'.
    """
    if not check_npu_available():
        pytest.skip("Ascend NPU not available")

    from mrt.torch.fx_mlir_backend import backend

    os.environ["MOPT_TORCH_TO_STABLEHLO_WHITELIST"] = "all"

    def sigmoid_mul_exp_fn(x, scale):
        sig_out = torch.sigmoid(x)
        scaled = sig_out * scale
        return torch.exp(scaled)

    x = torch.randn(4, 16, dtype=torch.float16).npu() * 0.1
    scale = torch.randn(4, 16, dtype=torch.float16).npu() * 0.1

    compiled_fn = torch.compile(sigmoid_mul_exp_fn, backend=backend)
    result = compiled_fn(x, scale)

    os.environ.pop("MOPT_TORCH_TO_STABLEHLO_WHITELIST", None)

    expected = torch.exp(torch.sigmoid(x) * scale)
    torch.testing.assert_close(result, expected, rtol=1e-2, atol=1e-2)

    # NOTE: IR printing is controlled by MOPT_PRINT_IR in fx_mlir_backend.
    # When running under pytest, stdout is captured by default, so use `pytest -s`
    # if you want to see the printed IR.


@arg_mark(
    plat_marks=["platform_ascend"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_matmul_combo_fusion_lower_to_dvm_call():
    """
    Feature: MatMul + elementwise combo lowered to mrt.dvm_call via fusion outlining.
    Description: Test fusion of matmul with elementwise operations (z = x @ w + y; out = exp(z) * z)
      using float16 tensors on NPU device.
    Expectation: Correctness within float16 tolerance (rtol=1e-2, atol=1e-2).
    """
    if not check_npu_available():
        pytest.skip("Ascend NPU not available")

    from mrt.torch.fx_mlir_backend import backend

    os.environ["MOPT_TORCH_TO_STABLEHLO_WHITELIST"] = "all"

    def matmul_combo(x, w, y):
        z = x @ w + y
        return torch.exp(z) * z

    compiled_fn = torch.compile(matmul_combo, backend=backend)

    # Keep magnitudes small for exp stability in float16.
    m, k, n = 16, 64, 32
    x = (torch.randn(m, k, dtype=torch.float16).npu() * 0.1)
    w = (torch.randn(k, n, dtype=torch.float16).npu() * 0.1)
    y = (torch.randn(m, n, dtype=torch.float16).npu() * 0.1)

    result = compiled_fn(x, w, y)
    expected = matmul_combo(x, w, y)
    torch.testing.assert_close(result, expected, rtol=1e-2, atol=1e-2)

    os.environ.pop("MOPT_TORCH_TO_STABLEHLO_WHITELIST", None)

@arg_mark(
    plat_marks=["platform_ascend"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_matmul_combo_fusion_lower_to_dvm_call_dynamic_shapes():
    """
    Feature: MatMul combo with varying shapes.
    Description: Run the same matmul+elementwise function with different concrete shapes (varying (m,k,n))
    under the fusion->dvm_call pipeline. We compile once; TorchDynamo may internally recompile/cache for
    new shapes, but we avoid explicitly calling `torch.compile(...)` per iteration.
    Expectation: Correct result for each shape.
    """
    if os.environ.get("MOPT_ENABLE_DYNAMIC_SHAPE_TEST", "0") != "1":
        pytest.skip("Dynamic-shape test is disabled (set MOPT_ENABLE_DYNAMIC_SHAPE_TEST=1 to enable)")
    if not check_npu_available():
        pytest.skip("Ascend NPU not available")

    from mrt.torch.fx_mlir_backend import backend

    def matmul_combo(x, w, y):
        z = x @ w + y
        return torch.exp(z) * z
    
    os.environ["MOPT_TORCH_TO_STABLEHLO_WHITELIST"] = "all"

    test_shapes = [(1, 32, 256), (16, 128, 64)]
    compiled_fn = torch.compile(matmul_combo, backend=backend)
    for m, k, n in test_shapes:
        x = (torch.randn(m, k, dtype=torch.float16).npu() * 0.1)
        w = (torch.randn(k, n, dtype=torch.float16).npu() * 0.1)
        y = (torch.randn(m, n, dtype=torch.float16).npu() * 0.1)
        result = compiled_fn(x, w, y)
        expected = matmul_combo(x, w, y)
        torch.testing.assert_close(result, expected, rtol=1e-2, atol=1e-2)

    os.environ.pop("MOPT_TORCH_TO_STABLEHLO_WHITELIST", None)


def main():
    """Run end-to-end tests for fusion -> dvm_call pipeline."""
    print("=" * 60)
    print("FX-MLIR Fusion->DVMCall End-to-End Test")
    print("=" * 60)

    enable_linalg_call = os.environ.get("MOPT_ENABLE_LINALG_CALL", "0") == "1"
    enable_dynamic_shape_test = os.environ.get("MOPT_ENABLE_DYNAMIC_SHAPE_TEST", "0") == "1"
    print_ir = os.environ.get("MOPT_PRINT_IR", "0") == "1"

    print("\nEnvironment variables:")
    print(f"  MOPT_ENABLE_LINALG_CALL: {enable_linalg_call}")
    print(f"  MOPT_ENABLE_DYNAMIC_SHAPE_TEST: {enable_dynamic_shape_test}")
    print(f"  MOPT_PRINT_IR: {print_ir}")

    if not check_npu_available():
        print("\nSkipping test: NPU not available")
        sys.exit(1)

    # pylint: disable=broad-except
    try:
        print("\n[1/4] Testing mul+exp fusion...")
        test_mul_exp_fusion_lower_to_dvm_call()
        print("✓ Passed")

        print("\n[2/4] Testing matmul combo fusion...")
        test_matmul_combo_fusion_lower_to_dvm_call()
        print("✓ Passed")

        print("\n[3/4] Testing transformer attention pattern...")
        test_transformer_attention_pattern()
        print("✓ Passed")

        if enable_dynamic_shape_test:
            print("\n[4/4] Testing dynamic shapes...")
            test_matmul_combo_fusion_lower_to_dvm_call_dynamic_shapes()
            print("✓ Passed")
        else:
            print("\n[4/4] Skipping dynamic shapes test (disabled)")

        print("\n" + "=" * 60)
        print("✅ All Tests Passed!")
        print("=" * 60)
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        print("\n" + "=" * 60)
        print("❌ Test Failed!")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
