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

"""Test StableHLO to Linalg conversion pass"""

import argparse
import os
import sys


def run_stablehlo_to_linalg_pass(mlir_text: str) -> str:
    """Run stablehlo-legalize-to-linalg pass using mopt Python API.

    Args:
        mlir_text: MLIR module text

    Returns:
        Converted MLIR module text
    """
    try:
        from mopt.passmanager import PassManager
        from mopt import ir
    except ImportError as e:
        print(f"Error: Failed to import mopt: {e}")
        print("Make sure mopt is installed:")
        print("  cd /home/lmy/inferrt/vllm-mindspore/mopt")
        print("  pip install -e .")
        sys.exit(1)

    # Create MLIR context and parse module
    ctx = ir.Context()
    mlir_module = ir.Module.parse(mlir_text, ctx)

    # Run the conversion pass
    with mlir_module.context:
        pm = PassManager.parse("builtin.module(stablehlo-legalize-to-linalg)")
        pm.run(mlir_module.operation)

    # Return converted module as text
    return str(mlir_module)


def check_conversion(output, expected_patterns, require_all=True):
    """Check if expected patterns are present in the output
    
    Args:
        output: The converted MLIR text
        expected_patterns: List of patterns to search for
        require_all: If True, all patterns must be present. If False, at least one must be present.
    
    Returns:
        List of missing patterns (empty if check passes)
    """
    if require_all:
        # All patterns must be present
        missing_patterns = []
        for pattern in expected_patterns:
            if pattern not in output:
                missing_patterns.append(pattern)
        return missing_patterns
    else:
        # At least one pattern must be present
        for pattern in expected_patterns:
            if pattern in output:
                return []  # Found at least one, success
        return expected_patterns  # None found, report all as missing


def test_stablehlo_to_linalg(dump_output=False):
    """
    Feature: StableHLO to Linalg conversion pass
    Description: Convert various StableHLO operations (add, multiply, matmul, reduce, etc.) to Linalg dialect operations
    Expectation: All test cases pass with correct Linalg operations generated for each StableHLO operation type
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_file = os.path.join(script_dir, 'stablehlo_to_linalg_test.mlir')

    if not os.path.exists(test_file):
        print(f"Error: Test file not found: {test_file}")
        sys.exit(1)

    print("=" * 80)
    print("Testing StableHLO to Linalg Conversion")
    print("=" * 80)
    print(f"Test file: {test_file}")
    print()

    # Read the MLIR test file
    with open(test_file, 'r', encoding='utf-8') as f:
        mlir_text = f.read()

    # Run the conversion pass using mopt Python API
    try:
        output = run_stablehlo_to_linalg_pass(mlir_text)
    except Exception as e:
        print(f"Error running conversion pass: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Check for key conversion patterns
    tests = [
        {
            'name': 'Element-wise Add',
            'patterns': ['linalg.generic', 'arith.addf'],
            'function': '@test_add'
        },
        {
            'name': 'Element-wise Multiply',
            'patterns': ['linalg.generic', 'arith.mulf'],
            'function': '@test_multiply'
        },
        {
            'name': 'Matrix Multiplication (dot)',
            'patterns': ['linalg.matmul'],
            'function': '@test_dot'
        },
        {
            'name': 'Batch Matrix Multiplication',
            'patterns': ['linalg.batch_matmul'],
            'function': '@test_dot_general'
        },
        {
            'name': 'Reduction (sum)',
            'patterns': ['linalg.reduce', 'linalg.generic'],  # May convert to either form
            'function': '@test_reduce_sum',
            'require_all': False  # Only need one pattern to match
        },
        {
            'name': 'Exponential',
            'patterns': ['linalg.generic', 'math.exp'],
            'function': '@test_exp'
        },
        {
            'name': 'Tanh',
            'patterns': ['linalg.generic', 'math.tanh'],
            'function': '@test_tanh'
        },
        {
            'name': 'Transpose',
            'patterns': ['linalg.transpose', 'linalg.generic'],  # May convert to either form
            'function': '@test_transpose',
            'require_all': False  # Only need one pattern to match
        },
        {
            'name': 'Convolution',
            'patterns': ['linalg.conv_2d_nhwc_hwcf'],  # NHWC canonical format
            'function': '@test_convolution'
        },
    ]

    passed = 0
    failed = 0
    results = []

    for test in tests:
        # Extract the function section from output
        func_start = output.find(f"func.func {test['function']}")
        if func_start == -1:
            results.append({
                'name': test['name'],
                'status': 'FAILED',
                'reason': f"Function {test['function']} not found in output"
            })
            failed += 1
            continue

        # Get the function body (up to next function or end)
        next_func = output.find("func.func @", func_start + 1)
        func_section = output[func_start:next_func] if next_func != -1 else output[func_start:]

        require_all = test.get('require_all', True)
        missing = check_conversion(func_section, test['patterns'], require_all)
        if missing:
            results.append({
                'name': test['name'],
                'status': 'FAILED',
                'reason': f"Missing patterns: {', '.join(missing)}"
            })
            failed += 1
        else:
            results.append({
                'name': test['name'],
                'status': 'PASSED',
                'reason': 'All patterns found'
            })
            passed += 1

    # Print results
    print("\nTest Results:")
    print("-" * 80)
    for result in results:
        status_symbol = "[OK]" if result['status'] == 'PASSED' else "[FAIL]"
        print(f"{status_symbol} {result['name']:<40} {result['status']:<10} {result['reason']}")

    print("-" * 80)
    print(f"Total: {len(tests)} tests, {passed} passed, {failed} failed")
    print("=" * 80)

    if failed > 0:
        print("\nSome tests failed. Please check the conversion implementation.")
        print("\nFull output for debugging:")
        print("-" * 80)
        print(output)
        sys.exit(1)
    else:
        print("\n[OK] All tests passed! StableHLO to Linalg conversion is working correctly.")
        if dump_output:
            print("\nConverted MLIR output:")
            print("-" * 80)
            print(output)
        return 0


def main():
    """Main entry point with argument parsing"""
    parser = argparse.ArgumentParser(
        description='Test StableHLO to Linalg conversion pass',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run tests normally (dump on failure only)
  python test_stablehlo_to_linalg.py
  
  # Always dump the converted output
  python test_stablehlo_to_linalg.py --dump
        """
    )
    
    parser.add_argument(
        '--dump',
        action='store_true',
        help='Always dump the converted MLIR output (default: False)'
    )
    
    args = parser.parse_args()
    
    return test_stablehlo_to_linalg(dump_output=args.dump)


if __name__ == '__main__':
    sys.exit(main())
