#!/usr/bin/env python3
"""
tests/st/inferrt/fusion/test_fusion_e2e.py

"""

import os
import sys
from tests.mark_utils import arg_mark

# Enable fusion pipeline
os.environ.setdefault("MOPT_ENABLE_FUSION", "1")

import torch


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
def test_mul_exp_fusion():
    """
    Feature: Sigmoid + Mul + Exp local fusion.
    Description: Run sigmoid(x) then fuse mul + exp on NPU using the fusion backend;
    inputs are random fp16 tensors (x, scale) on a single card.
    Expectation: Compiled fusion output matches eager torch within rtol/atol 1e-2.

    Network structure:
        x ─ sigmoid ─┬─ mul ─ exp ─ output
        scale ───────┘

    Computation flow:
        sig_out = sigmoid(x)      # Not in whitelist → Torch → mrt.sigmoid (step 6)
        scaled = sig_out * scale  # In whitelist → StableHLO } fused
        result = exp(scaled)      # In whitelist → StableHLO } into single linalg_call
    """
    from mrt.torch.fx_mlir_backend import backend

    print("\n" + "=" * 60)
    print("Test: Sigmoid + Mul + Exp Local Fusion")
    print("=" * 60)

    def sigmoid_mul_exp_fn(x, scale):
        # Sigmoid - not in whitelist, stays as Torch, converted to mrt.sigmoid
        sig_out = torch.sigmoid(x)
        # Mul + Exp - in whitelist, converted to StableHLO and fused into single linalg_call
        scaled = sig_out * scale
        result = torch.exp(scaled)
        return result

    # Compile with fusion backend
    compiled_fn = torch.compile(sigmoid_mul_exp_fn, backend=backend)

    # Test inputs (small values to avoid exp overflow)
    x = torch.randn(4, 16, dtype=torch.float16).npu() * 0.1
    scale = torch.randn(4, 16, dtype=torch.float16).npu() * 0.1

    # Run compiled function
    result = compiled_fn(x, scale)

    # Calculate reference result
    sig_out = torch.sigmoid(x)
    expected = torch.exp(sig_out * scale)

    # Verify correctness
    max_diff = torch.max(torch.abs(result - expected)).item()
    passed = torch.allclose(result, expected, rtol=1e-2, atol=1e-2)

    print(f"  Input tensor shapes: x={x.shape}, scale={scale.shape}")
    print(f"  Output tensor shape: {result.shape}")
    print(f"  Max difference: {max_diff:.6e}")
    print(f"  Result: {'Passed' if passed else 'Failed'}")

    assert passed, f"Local fusion test failed, max_diff={max_diff}"
    return True


def main():
    """Run end-to-end tests for fusion pipeline."""
    print("=" * 60)
    print("FX-MLIR Fusion Pipeline End-to-End Test")
    print("=" * 60)

    # Print environment configuration
    enable_fusion = os.environ.get("MOPT_ENABLE_FUSION", "0") == "1"
    print_ir = os.environ.get("MOPT_PRINT_IR", "0") == "1"

    print("\nEnvironment variables:")
    print(f"  MOPT_ENABLE_FUSION: {enable_fusion}")
    print(f"  MOPT_PRINT_IR: {print_ir}")

    # Check NPU availability
    if not check_npu_available():
        print("\nSkipping test: NPU not available")
        sys.exit(1)

    # Run tests
    # pylint: disable=broad-except
    try:
        test_mul_exp_fusion()
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
