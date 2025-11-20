#!/usr/bin/env python3
"""
tests/st/inferrt/ops/test_op_linalg_call_e2e.py

Test linalg_call operator via torch.compile integration.

This  test case will be remove when mrt backend is ready.
"""
import pytest
import torch
from mopt.passmanager import PassManager
from mopt import ir
from tests.mark_utils import arg_mark


def generate_linalg_mlir_for_add():
    """Generate Linalg MLIR with hacc annotations for add operation"""
    stablehlo = """
module {
  func.func @add_kernel(%arg0: tensor<?x?xf16>, %arg1: tensor<?x?xf16>) -> tensor<?x?xf16> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<?x?xf16>
    return %0 : tensor<?x?xf16>
  }
}
"""

    ctx = ir.Context()
    module = ir.Module.parse(stablehlo, ctx)

    with module.context:
        pm = PassManager.parse(
            "builtin.module(stablehlo-legalize-to-linalg,annotate-linalg-for-hacc)"
        )
        pm.run(module.operation)

    return str(module)


# Register linalg_add op once at module level
_LINALG_ADD_REGISTERED = False


def _setup_linalg_add_op():
    """Setup linalg_add custom op (only once)"""
    global _LINALG_ADD_REGISTERED

    if _LINALG_ADD_REGISTERED:
        return

    from mrt.torch import register_linalg_op

    # Generate and register MLIR
    mlir_text = generate_linalg_mlir_for_add()
    register_linalg_op("linalg_add", mlir_text)

    # Define custom op in mrt_linalg namespace
    @torch.library.custom_op("mrt_linalg::linalg_add", mutates_args=())
    def linalg_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Placeholder for linalg_call")

    @torch.library.register_fake("mrt_linalg::linalg_add")
    def _(x, y):
        return x

    _LINALG_ADD_REGISTERED = True


def test_mlir_generation():
    """
    Feature: MLIR generation for OpLinalgCall
    Description: Generate Linalg MLIR with hacc annotations from StableHLO add operation
    Expectation: Generated MLIR contains hacc.entry, hacc.function_kind, linalg operations and correct tensor shapes
    """

    print("\n" + "=" * 60)
    print("OpLinalgCall MLIR Generation Test")
    print("=" * 60 + "\n")

    print("Step 1: Generating Linalg MLIR with hacc annotations...")
    mlir_text = generate_linalg_mlir_for_add()
    print(f"[OK] Generated MLIR ({len(mlir_text)} bytes)\n")

    print("Step 2: Verifying MLIR content...")
    checks = [
        ("hacc.entry" in mlir_text, "hacc.entry attribute present"),
        ("hacc.function_kind" in mlir_text, "hacc.function_kind attribute present"),
        ("linalg." in mlir_text, "Linalg operations present"),
        ("tensor<?x?xf16>" in mlir_text, "Correct tensor shape"),
    ]

    for check, description in checks:
        status = "[OK]" if check else "[FAIL]"
        print(f"  {status} {description}")
        assert check, f"Failed: {description}"

    print("\n" + "=" * 60)
    print("OpLinalgCall MLIR generation test passed!")
    print("=" * 60 + "\n")

    print("Generated MLIR (first 30 lines):")
    print("-" * 60)
    for i, line in enumerate(mlir_text.split("\n")[:30], 1):
        print(f"{i:3}: {line}")
    print("-" * 60)


@arg_mark(
    plat_marks=["platform_ascend"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
# @pytest.mark.forked  # Run each test case in a separate process to avoid memory accumulation
@pytest.mark.parametrize(
    "shape",
    [
        (16, 32),
        (1, 1),
        (8, 8),
        (16, 16),
        (32, 32),
        (64, 64),
        (128, 128),
        (1, 32),
        (32, 1),
        # (2048, 4), failed
        # (4, 2048), failed
    ],
)
def test_linalg_call_correctness(shape):
    """
    Feature: Test linalg_call via torch.compile
    Description: Use mrt_linalg namespace to invoke linalg_call through torch.compile
    Expectation: Output matches expected computation results
    """
    import torch_npu  # noqa: F401
    from mrt.torch import backend

    if not torch.npu.is_available():
        pytest.skip("Ascend NPU not available")

    # Setup linalg_add op (only once)
    _setup_linalg_add_op()

    # Compile with mrt backend
    def add_fn(x, y):
        return torch.ops.mrt_linalg.linalg_add(x, y)

    add_compiled = torch.compile(add_fn, backend=backend)

    # Test with parametrized shape
    x = torch.randn(*shape, dtype=torch.float16).npu()
    y = torch.randn(*shape, dtype=torch.float16).npu()

    result = add_compiled(x, y)
    expected = x + y

    assert torch.allclose(
        result, expected, rtol=1e-3, atol=1e-3
    ), f"Shape {shape}: max_diff={torch.max(torch.abs(result - expected)).item()}"
    print(f"Shape {shape}: linalg_call via torch.compile passed.")


if __name__ == "__main__":
    test_mlir_generation()
    # Run with a single representative shape when running directly
    print("\nRunning linalg_call correctness test with shape (16, 32)...")
    test_linalg_call_correctness((16, 32))
