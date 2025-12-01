"""
Test for annotate-linalg-for-hacc pass.

This test verifies that the pass correctly adds hacc attributes to functions
containing Linalg operations.
"""

import os
import sys
from pathlib import Path

# Use installed mopt package (installed via pip install mopt/dist/*.whl)
# No need to manipulate sys.path - just import directly
from mopt.passmanager import PassManager
from mopt import ir


def test_annotate_linalg_pass():
    """
    Feature: annotate-linalg-for-hacc pass
    Description: Apply hacc attributes to functions containing Linalg operations while leaving functions without Linalg ops unannotated
    Expectation: Functions with Linalg ops get hacc.entry and hacc.function_kind attributes, functions without Linalg ops remain unannotated
    """

    # Test input MLIR
    mlir_text = """
    module {
      func.func @add_kernel(%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>) -> tensor<4x8xf32> {
        %0 = tensor.empty() : tensor<4x8xf32>
        %1 = linalg.generic {
          indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                           affine_map<(d0, d1) -> (d0, d1)>,
                           affine_map<(d0, d1) -> (d0, d1)>],
          iterator_types = ["parallel", "parallel"]
        } ins(%arg0, %arg1 : tensor<4x8xf32>, tensor<4x8xf32>)
          outs(%0 : tensor<4x8xf32>) {
        ^bb0(%in0: f32, %in1: f32, %out: f32):
          %2 = arith.addf %in0, %in1 : f32
          linalg.yield %2 : f32
        } -> tensor<4x8xf32>
        return %1 : tensor<4x8xf32>
      }
      
      func.func @helper(%arg0: f32) -> f32 {
        %0 = arith.mulf %arg0, %arg0 : f32
        return %0 : f32
      }
    }
    """
    
    # Parse MLIR
    ctx = ir.Context()
    module = ir.Module.parse(mlir_text, ctx)
    
    # Run the annotate pass
    with module.context:
        pm = PassManager.parse("builtin.module(annotate-linalg-for-hacc)")
        pm.run(module.operation)
    
    # Get output
    output = str(module)
    
    # Verify annotations
    assert 'hacc.entry' in output, "hacc.entry attribute not found"
    assert 'hacc.function_kind' in output, "hacc.function_kind attribute not found"
    assert '#hacc.function_kind<HOST>' in output, "HOST function kind not found"
    
    # Verify that add_kernel is annotated
    assert '@add_kernel' in output, "add_kernel function not found"
    
    # Verify that helper function (no Linalg ops) is not over-annotated
    lines = output.split('\n')
    add_kernel_attrs = 0
    helper_attrs = 0
    
    for i, line in enumerate(lines):
        if '@add_kernel' in line:
            # Check if hacc.entry is in the same line or nearby
            for j in range(max(0, i-2), min(len(lines), i+3)):
                if 'hacc.entry' in lines[j]:
                    add_kernel_attrs += 1
                    break
        elif '@helper' in line:
            # Helper should NOT have hacc.entry
            for j in range(max(0, i-2), min(len(lines), i+3)):
                if 'hacc.entry' in lines[j]:
                    helper_attrs += 1
                    break
    
    assert add_kernel_attrs > 0, "add_kernel should be annotated"
    assert helper_attrs == 0, "helper should NOT be annotated (no Linalg ops)"
    
    print("[OK] Pass test passed")
    print(f"\nAnnotated MLIR:\n{output}")


def test_stablehlo_to_linalg_with_annotation():
    """
    Feature: StableHLO to Linalg conversion with hacc annotation pipeline
    Description: Convert StableHLO add operation to Linalg dialect and apply hacc annotations in a single pipeline
    Expectation: Output contains Linalg operations with hacc.entry and hacc.function_kind attributes
    """

    stablehlo_mlir = """
    module {
      func.func @add(%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>) -> tensor<4x8xf32> {
        %0 = stablehlo.add %arg0, %arg1 : tensor<4x8xf32>
        return %0 : tensor<4x8xf32>
      }
    }
    """

    # Parse MLIR
    ctx = ir.Context()
    module = ir.Module.parse(stablehlo_mlir, ctx)

    # Run full pipeline
    with module.context:
        pm = PassManager.parse(
            "builtin.module("
            "stablehlo-legalize-to-linalg,"
            "annotate-linalg-for-hacc"
            ")"
        )
        pm.run(module.operation)

    # Get output
    output = str(module)

    # Verify conversion and annotation
    assert 'linalg.generic' in output or 'linalg.' in output, "Linalg operations not found"
    assert 'hacc.entry' in output, "hacc.entry attribute not found"
    assert 'hacc.function_kind' in output, "hacc.function_kind attribute not found"

    print("[OK] StableHLO -> Linalg -> hacc annotation pipeline passed")
    print(f"\nFinal MLIR:\n{output}")


def test_idempotency():
    """
    Feature: annotate-linalg-for-hacc pass idempotency
    Description: Run annotate-linalg-for-hacc pass twice on the same MLIR module
    Expectation: Running the pass twice produces identical results without duplicating attributes
    """

    mlir_text = """
    module {
      func.func @kernel(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> {
        %0 = tensor.empty() : tensor<4x8xf32>
        %cst = arith.constant 0.0 : f32
        %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<4x8xf32>) -> tensor<4x8xf32>
        return %1 : tensor<4x8xf32>
      }
    }
    """
    
    # Parse MLIR
    ctx = ir.Context()
    module = ir.Module.parse(mlir_text, ctx)
    
    # Run pass twice
    with module.context:
        pm1 = PassManager.parse("builtin.module(annotate-linalg-for-hacc)")
        pm1.run(module.operation)
        
        output1 = str(module)
        
        pm2 = PassManager.parse("builtin.module(annotate-linalg-for-hacc)")
        pm2.run(module.operation)
        
        output2 = str(module)
    
    # Count occurrences of hacc.entry
    count1 = output1.count('hacc.entry')
    count2 = output2.count('hacc.entry')
    
    assert count1 == count2, f"Pass is not idempotent: {count1} vs {count2} occurrences"
    assert count1 > 0, "No annotations found"
    
    print("[OK] Idempotency test passed")


if __name__ == "__main__":
    print("Running annotate-linalg-for-hacc tests...\n")
    
    try:
        test_annotate_linalg_pass()
        print("\n" + "="*60 + "\n")
        
        test_stablehlo_to_linalg_with_annotation()
        print("\n" + "="*60 + "\n")
        
        test_idempotency()
        print("\n" + "="*60 + "\n")
        
        print("\n[PASS] All tests passed!")
        
    except Exception as e:
        print(f"\n[FAIL] Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
