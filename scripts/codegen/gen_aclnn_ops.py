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

"""Generator for Aclnn ops launch code."""

from typing import List, Dict, Any

from utils import format_directory, save_file
import constants as C
from code_generator import CodeGenerator


def _convert_type(obj, obj_type, optional_mapping: Dict[str, str] = None) -> str:
    """Convert mrt dialect type to mrt ir type."""
    if obj_type == 'Mrt_ScalarType':
        # the ValuePtr will be converted to aclScalar
        return obj

    pure_types_map = {
        'Mrt_DtypeType': f'static_cast<mrt::ir::DataType::Type>({obj}->ToInt())',
        'Mrt_I64Type': f'{obj}->ToInt()',
        'Mrt_BooleanType': f'{obj}->ToBool()',
        'Mrt_F32Type': f'{obj}->ToFloat()',
        'Mrt_F64Type': f'{obj}->ToDouble()',
        'Mrt_StringType': f'{obj}->ToString()',
    }
    if obj_type in pure_types_map:
        return pure_types_map[obj_type]

    ptr_types_map = {
        'MrtAnyTensor': (f'{obj}->ToTensor()', f'{obj}->IsTensor()'),
        'Mrt_I64ArrayType' : (f'{obj}->ToTuple()', f'{obj}->IsTuple()'),
        'Mrt_BooleanArrayType' : (f'{obj}->ToTuple()', f'{obj}->IsTuple()'),
        'Mrt_F32ArrayType' : (f'{obj}->ToTuple()', f'{obj}->IsTuple()'),
        'Mrt_F64ArrayType' : (f'{obj}->ToTuple()', f'{obj}->IsTuple()'),
        'MrtTensorList' : (f'{obj}->ToTuple()', f'{obj}->IsTuple()'),
    }
    if obj_type in ptr_types_map:
        return ptr_types_map[obj_type][0]
    if optional_mapping is not None and obj_type in optional_mapping and optional_mapping[obj_type] in ptr_types_map:
        v = ptr_types_map[optional_mapping[obj_type]]
        return f'{v[1]} ? {v[0]} : nullptr'
    raise ValueError(f"Invalid argument type: {obj_type}")

class AclnnOpsGenerator(CodeGenerator):
    """Generator for Aclnn ops launch code."""
    def __init__(self, optional_mapping: Dict[str, str] = None):
        """
        Initialize the generator.

        Args:
            optional_mapping: Dictionary mapping anonymous Optional types to their baseType.def values.
        """
        super().__init__([C.COMMON_TEMPLATES_DIR, C.ACLNN_OPS_TEMPLATES_DIR], C.ACLNN_OPS_AUTO_GEN_DIR)

        # Store optional mapping for resolving anonymous types
        self.optional_mapping = optional_mapping or {}

        # Filters for inputs and outputs processing of aclnn ops
        self.gen_env.filters['arg_wrapper'] = self._arg_wrapper
        self.gen_env.filters['output_wrapper'] = self._output_wrapper
        self.gen_env.filters['is_output_in_ioref'] = self._is_output_in_ioref

    # pylint: disable=arguments-differ
    def generate(self, ops_defs: List[List[dict[str, Any]]]) -> None:
        """Generate C++ launch code for all aclnn ops."""
        for ops_def in ops_defs:
            for op_def in ops_def:
                self._generate_header(op_def)
                self._generate_source(op_def)
        print(f"Generated aclnn ops launch code in {self.output_dir} success")

        # Apply clang-format to the generated code if available
        format_directory(self.output_dir)

    def _generate_header(self, op_def: Dict[str, Any]) -> None:
        """Generate header for an aclnn op."""
        template = self.gen_env.get_template('aclnn_op.h.j2')
        ioref_pairs = op_def.get('iorefPairs', [])
        template_data = {
            'op_name': op_def['opName'],
            'gen_aclnn_op': op_def['genAclnnOp'],
            'has_ioref': len(ioref_pairs) > 0,
        }
        # Initialize _current_ioref_pairs for safety (though header template doesn't use output_wrapper)
        self._current_ioref_pairs = ioref_pairs
        generated_str = template.render(**template_data)
        save_file(generated_str, self.output_dir, f"{self._to_snake_case(op_def['opName'])}.h")

    def _generate_source(self, op_def: Dict[str, Any]) -> None:
        """Generate source for an aclnn op."""
        template = self.gen_env.get_template('aclnn_op.cc.j2')
        ioref_pairs = op_def.get('iorefPairs', [])
        template_data = {
            'op_name': op_def['opName'],
            'gen_aclnn_op': op_def['genAclnnOp'],
            'args': op_def['arguments']['args'],
            'result': op_def['results']['args'],
            'ioref_pairs': ioref_pairs,
            'has_ioref': len(ioref_pairs) > 0,
        }
        # Store ioref_pairs in the generator instance for use in filters
        self._current_ioref_pairs = ioref_pairs
        generated_str = template.render(**template_data)
        save_file(generated_str, self.output_dir, f"{self._to_snake_case(op_def['opName'])}.cc")

    def _arg_wrapper(self, args: list) -> str:
        """Wrapper for arguments."""
        ret = ''
        for idx, arg in enumerate(args):
            ret += ", " + _convert_type(f"input[{idx}]", arg[0].get('def'), self.optional_mapping)
        return ret

    def _output_wrapper(self, result: list) -> str:
        """Wrapper for results. Skips outputs that are in IORef."""
        # Get IORef pairs from instance variable (set in _generate_source)
        ioref_pairs = getattr(self, '_current_ioref_pairs', [])

        # Build a set of output indices that are in IORef
        ioref_output_indices = {pair[0] for pair in ioref_pairs}

        if len(result) > 1:
            ret = ''
            for idx, result_item in enumerate(result):
                # Skip outputs that are in IORef
                if idx in ioref_output_indices:
                    continue
                result_item_type = result_item[0].get('def')
                ret += ", " + _convert_type(f"(*output_tuple)[{idx}]", result_item_type)
            return ret

        # Single output case
        if 0 in ioref_output_indices:
            # Output 0 is in IORef, skip it
            return ""
        return ", " + _convert_type("output", result[0][0].get('def'))

    def _is_output_in_ioref(self, output_idx: int, ioref_pairs: list) -> bool:
        """Check if an output index is in IORef pairs."""
        if not ioref_pairs:
            return False
        return output_idx in {pair[0] for pair in ioref_pairs}
