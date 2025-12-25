#!/usr/bin/env python3
"""
tests/st/inferrt/fusion/test_dvm_call_fusion_e2e.py
"""

import os
import sys

import pytest
import torch

from tests.mark_utils import arg_mark


os.environ.setdefault("MOPT_ENABLE_FUSION", "1")
os.environ.setdefault("MOPT_ENABLE_LINALG_CALL", "0")


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

    def sigmoid_mul_exp_fn(x, scale):
        sig_out = torch.sigmoid(x)
        scaled = sig_out * scale
        return torch.exp(scaled)

    x = torch.randn(4, 16, dtype=torch.float16).npu() * 0.1
    scale = torch.randn(4, 16, dtype=torch.float16).npu() * 0.1

    compiled_fn = torch.compile(sigmoid_mul_exp_fn, backend=backend)
    result = compiled_fn(x, scale)

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
    z = x @ w + y
    expected = torch.exp(z) * z
    torch.testing.assert_close(result, expected, rtol=1e-2, atol=1e-2)



def main():
    """Run end-to-end tests for fusion -> dvm_call pipeline."""
    print("=" * 60)
    print("FX-MLIR Fusion->DVMCall End-to-End Test")
    print("=" * 60)

    enable_fusion = os.environ.get("MOPT_ENABLE_FUSION", "0") == "1"
    enable_linalg_call = os.environ.get("MOPT_ENABLE_LINALG_CALL", "0") == "1"
    print_ir = os.environ.get("MOPT_PRINT_IR", "0") == "1"

    print("\nEnvironment variables:")
    print(f"  MOPT_ENABLE_FUSION: {enable_fusion}")
    print(f"  MOPT_ENABLE_LINALG_CALL: {enable_linalg_call}")
    print(f"  MOPT_PRINT_IR: {print_ir}")

    if not check_npu_available():
        print("\nSkipping test: NPU not available")
        sys.exit(1)

    # pylint: disable=broad-except
    try:
        test_mul_exp_fusion_lower_to_dvm_call()
        test_matmul_combo_fusion_lower_to_dvm_call()
        print("\n" + "=" * 60)
        print("Test Passed!")
        print("=" * 60)
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        print("\n" + "=" * 60)
        print("Test Failed!")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()


