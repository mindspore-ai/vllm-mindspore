# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Test StableHLO to DVM conversion pass"""

import argparse
import os
import sys


def run_stablehlo_to_dvm_pass(mlir_text: str) -> str:
    """Run convert-stablehlo-to-dvm pass using mopt Python API."""
    try:
        from mopt.passmanager import PassManager
        from mopt import ir
    except ImportError as e:
        print(f"Error: Failed to import mopt: {e}")
        print("Make sure mopt is installed.")
        sys.exit(1)

    # Create MLIR context and parse module
    ctx = ir.Context()
    mlir_module = ir.Module.parse(mlir_text, ctx)

    # Run the conversion pass
    with mlir_module.context:
        pm = PassManager.parse("builtin.module(convert-stablehlo-to-dvm)")
        pm.run(mlir_module.operation)

    # Return converted module as text
    return str(mlir_module)


def check_conversion(output, expected_patterns, forbidden_patterns=None):
    """Check if expected patterns are present and forbidden patterns are absent in the output"""
    errors = []
    for pattern in expected_patterns:
        if pattern not in output:
            errors.append(f"Missing expected pattern: {pattern}")

    if forbidden_patterns:
        for pattern in forbidden_patterns:
            if pattern in output:
                errors.append(f"Found forbidden pattern: {pattern}")

    return errors


def test_stablehlo_to_dvm(dump_output=False):
    """
    Feature: StableHLO to DVM conversion pass
    Description: Convert StableHLO unary operations to Dvm operations
    Expectation: All StableHLO to DVM conversion patterns are present in the output
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_file = os.path.join(script_dir, 'mlir', 'stablehlo_to_dvm_test.mlir')

    if not os.path.exists(test_file):
        print(f"Error: Test file not found: {test_file}")
        sys.exit(1)

    print("=" * 80)
    print("Testing StableHLO to DVM Conversion")
    print("=" * 80)
    print(f"Test file: {test_file}")

    # Read the MLIR test file
    with open(test_file, 'r', encoding='utf-8') as f:
        mlir_text = f.read()

    # Run the conversion pass
    try:
        output = run_stablehlo_to_dvm_pass(mlir_text)
    except Exception as e:
        print(f"Error running conversion pass: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    # Check for key conversion patterns
    # Note: We expect dvm.unary op_type format
    # The op_type values correspond to I32EnumAttrCase definitions in DvmOps.td
    tests = [
        {'name': 'Sqrt', 'function': '@test_sqrt', 'patterns': ['dvm.load', 'dvm.unary Sqrt', 'dvm.store']},
        {'name': 'Abs', 'function': '@test_abs', 'patterns': ['dvm.load', 'dvm.unary Abs', 'dvm.store']},
        {'name': 'Exp', 'function': '@test_exp', 'patterns': ['dvm.load', 'dvm.unary Exp', 'dvm.store']},
        {'name': 'Log', 'function': '@test_log', 'patterns': ['dvm.load', 'dvm.unary Log', 'dvm.store']},
        {'name': 'IsFinite', 'function': '@test_isfinite', 'patterns': ['dvm.load', 'dvm.unary IsFinite', 'dvm.store']},
        {'name': 'LogicalNot', 'function': '@test_not', 'patterns': ['dvm.load', 'dvm.unary LogicalNot', 'dvm.store']},
        {'name': 'Round', 'function': '@test_round', 'patterns': ['dvm.load', 'dvm.unary Round', 'dvm.store']},
        {'name': 'Floor', 'function': '@test_floor', 'patterns': ['dvm.load', 'dvm.unary Floor', 'dvm.store']},
        {'name': 'Ceil', 'function': '@test_ceil', 'patterns': ['dvm.load', 'dvm.unary Ceil', 'dvm.store']},
        # Binary Ops
        {'name': 'Add', 'function': '@test_add', 'patterns': ['dvm.load', 'dvm.binary Add', 'dvm.store']},
        {'name': 'Sub', 'function': '@test_sub', 'patterns': ['dvm.load', 'dvm.binary Sub', 'dvm.store']},
        {'name': 'Mul', 'function': '@test_mul', 'patterns': ['dvm.load', 'dvm.binary Mul', 'dvm.store']},
        {'name': 'Div', 'function': '@test_div', 'patterns': ['dvm.load', 'dvm.binary Div', 'dvm.store']},
        {'name': 'Pow', 'function': '@test_pow', 'patterns': ['dvm.load', 'dvm.binary Pow', 'dvm.store']},
        {'name': 'Maximum', 'function': '@test_max', 'patterns': ['dvm.load', 'dvm.binary Maximum', 'dvm.store']},
        {'name': 'Minimum', 'function': '@test_min', 'patterns': ['dvm.load', 'dvm.binary Minimum', 'dvm.store']},
        {'name': 'LogicalAnd', 'function': '@test_and', 'patterns': ['dvm.load', 'dvm.binary LogicalAnd', 'dvm.store']},
        {'name': 'LogicalOr', 'function': '@test_or', 'patterns': ['dvm.load', 'dvm.binary LogicalOr', 'dvm.store']},
        {'name': 'Equal', 'function': '@test_eq', 'patterns': ['dvm.load', 'dvm.binary Equal', 'dvm.store']},
        {'name': 'NotEqual', 'function': '@test_ne', 'patterns': ['dvm.load', 'dvm.binary NotEqual', 'dvm.store']},
        {'name': 'Greater', 'function': '@test_gt', 'patterns': ['dvm.load', 'dvm.binary Greater', 'dvm.store']},
        {
            'name': 'GreaterEqual',
            'function': '@test_ge',
            'patterns': ['dvm.load', 'dvm.binary GreaterEqual', 'dvm.store'],
        },
        {'name': 'Less', 'function': '@test_lt', 'patterns': ['dvm.load', 'dvm.binary Less', 'dvm.store']},
        {'name': 'LessEqual', 'function': '@test_le', 'patterns': ['dvm.load', 'dvm.binary LessEqual', 'dvm.store']},
        # Dot Ops
        {
            'name': 'Dot 2D',
            'function': '@test_dot_2d',
            'patterns': ['dvm.load', 'dvm.matmul', 'trans_a false', 'trans_b false', 'dvm.store'],
        },
        {
            'name': 'Dot Vector-Vector',
            'function': '@test_dot_vector_vector',
            'patterns': ['dvm.load', 'dvm.matmul', 'dvm.reshape', 'dvm.store'],
        },
        {
            'name': 'Dot Matrix-Vector',
            'function': '@test_dot_matrix_vector',
            'patterns': ['dvm.load', 'dvm.matmul', 'dvm.reshape', 'dvm.store'],
        },
        {
            'name': 'Dot Vector-Matrix',
            'function': '@test_dot_vector_matrix',
            'patterns': ['dvm.load', 'dvm.matmul', 'dvm.reshape', 'dvm.store'],
        },
        {
            'name': 'DotGeneral Standard',
            'function': '@test_dot_general_standard',
            'patterns': ['dvm.load', 'dvm.matmul', 'trans_a false', 'trans_b false', 'dvm.store'],
        },
        {
            'name': 'DotGeneral Transposed',
            'function': '@test_dot_general_transposed',
            'patterns': ['dvm.load', 'dvm.matmul', 'trans_a true', 'trans_b true', 'dvm.store'],
        },
        {
            'name': 'DotGeneral Multi-Dim Batch',
            'function': '@test_dot_general_multi_dim',
            'patterns': ['dvm.load', 'dvm.matmul', 'trans_a false', 'trans_b false', 'dvm.store'],
        },
        {
            'name': 'DotGeneral Flatten M',
            'function': '@test_dot_general_flatten_M',
            'patterns': ['dvm.load', 'dvm.reshape', 'dvm.matmul', 'dvm.reshape', 'dvm.store'],
        },
        {
            'name': 'DotGeneral Flatten N',
            'function': '@test_dot_general_flatten_N',
            'patterns': ['dvm.load', 'dvm.reshape', 'dvm.matmul', 'dvm.reshape', 'dvm.store'],
        },
        {
            'name': 'DotGeneral Flatten K',
            'function': '@test_dot_general_flatten_K',
            'patterns': ['dvm.load', 'dvm.reshape', 'dvm.matmul', 'dvm.store'],
        },
        {
            'name': 'DotGeneral Transposed Flatten',
            'function': '@test_dot_general_transposed_flatten',
            'patterns': ['dvm.load', 'dvm.reshape', 'dvm.matmul', 'dvm.store'],
        },
        # Constant Lifting
        {
            'name': 'Lift Constant Callee',
            'function': '@test_lift_constant_callee',
            'patterns': ['dvm.load', 'dvm.load', 'dvm.binary Add', 'dvm.store'],
            'forbidden_patterns': ['stablehlo.constant'],
        },
        {
            'name': 'Lift Constant Caller',
            'function': '@test_lift_constant_caller',
            'patterns': ['stablehlo.constant', 'call @test_lift_constant_callee', 'stablehlo.add'],
            'forbidden_patterns': ['dvm.binary Add', 'dvm.load', 'dvm.store'],
        },
    ]

    passed = 0
    failed = 0
    results = []

    for test in tests:
        # Extract the function section from output
        func_start = output.find(f"func.func {test['function']}")
        if func_start == -1:
            results.append(
                {'name': test['name'], 'status': 'FAILED', 'reason': f"Function {test['function']} not found in output"}
            )
            failed += 1
            continue

        # Get the function body
        next_func = output.find("func.func @", func_start + 1)
        func_section = output[func_start:next_func] if next_func != -1 else output[func_start:]

        forbidden = test.get('forbidden_patterns', [])
        errors = check_conversion(func_section, test['patterns'], forbidden)
        if errors:
            results.append({'name': test['name'], 'status': 'FAILED', 'reason': '; '.join(errors)})
            failed += 1
        else:
            results.append({'name': test['name'], 'status': 'PASSED', 'reason': 'All patterns found'})
            passed += 1

    # Print results
    print("\nTest Results:")
    print("-" * 80)
    for result in results:
        status_symbol = "[OK]" if result['status'] == 'PASSED' else "[FAIL]"
        print(f"{status_symbol} {result['name']:<20} {result['status']:<10} {result['reason']}")

    print("-" * 80)
    print(f"Total: {len(tests)} tests, {passed} passed, {failed} failed")
    print("=" * 80)

    if failed > 0:
        print("\nSome tests failed. Please check the conversion implementation.")
        if dump_output:
            print("\nFull output for debugging:")
            print("-" * 80)
            print(output)
        sys.exit(1)
    else:
        print("\n[OK] All tests passed! StableHLO to DVM conversion is working correctly.")
        if dump_output:
            print("\nConverted MLIR output:")
            print("-" * 80)
            print(output)
        return 0


def main():
    """Main entry point with argument parsing"""
    parser = argparse.ArgumentParser(
        description='Test StableHLO to DVM conversion pass', formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--dump', action='store_true', help='Always dump the converted MLIR output (default: False)')

    args = parser.parse_args()

    return test_stablehlo_to_dvm(dump_output=args.dump)


if __name__ == '__main__':
    sys.exit(main())
