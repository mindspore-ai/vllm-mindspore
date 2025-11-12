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


def _convert_type(input_type: str) -> str:
    """Convert mrt dialect type to mrt ir type."""
    types_map = {
        'AnyRankedTensor': '->ToTensor()',
        'I64Attr': '->ToInt()',
        'BoolAttr': '->ToBool()',
        'F32Attr': '->ToFloat()',
        'F64Attr': '->ToDouble()',
    }
    tuple_types_set = {
        'I64ArrayAttr',
        'BoolArrayAttr',
        'F32ArrayAttr',
        'F64ArrayAttr',
        'MrtTensorList',
    }

    if input_type in types_map:
        return types_map[input_type]
    if input_type not in tuple_types_set:
        raise ValueError(f"Invalid argument type: {input_type}")
    return '->ToTuple()'

class AclnnOpsGenerator(CodeGenerator):
    """Generator for Aclnn ops launch code."""
    def __init__(self):
        """
        Initialize the generator.
        """
        super().__init__([C.COMMON_TEMPLATES_DIR, C.ACLNN_OPS_TEMPLATES_DIR], C.ACLNN_OPS_AUTO_GEN_DIR)

        # Filters for inputs and outputs processing of aclnn ops
        self.gen_env.filters['arg_wrapper'] = self._arg_wrapper
        self.gen_env.filters['output_wrapper'] = self._output_wrapper

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
        template_data = {
            'op_name': op_def['opName'],
        }
        generated_str = template.render(**template_data)
        save_file(generated_str, self.output_dir, f"{self._to_snake_case(op_def['opName'])}.h")

    def _generate_source(self, op_def: Dict[str, Any]) -> None:
        """Generate source for an aclnn op."""
        template = self.gen_env.get_template('aclnn_op.cc.j2')
        template_data = {
            'op_name': op_def['opName'],
            'args': op_def['arguments']['args'],
            'result': op_def['results']['args']
        }
        generated_str = template.render(**template_data)
        save_file(generated_str, self.output_dir, f"{self._to_snake_case(op_def['opName'])}.cc")

    def _arg_wrapper(self, args: list) -> str:
        """Wrapper for arguments."""
        ret = ''
        for idx, arg in enumerate(args):
            arg_type = arg[0].get('def')
            if arg_type == 'MrtOptTensor':
                ret += ''.join(f", input[kIndex{idx}]->IsTensor() ? input[kIndex{idx}]->ToTensor() : nullptr")
            else:
                ret += ''.join(f", input[kIndex{idx}]{_convert_type(arg_type)}")
        return ret

    def _output_wrapper(self, result: list) -> str:
        """Wrapper for results."""
        if len(result) > 1:
            ret = ''
            for idx, result_item in enumerate(result):
                result_item_type = result_item[0].get('def')
                ret += ''.join(f", (*output_tuple)[kIndex{idx}]{_convert_type(result_item_type)}")
            return ret
        return ''.join(f", output{_convert_type(result[0][0].get('def'))}")
