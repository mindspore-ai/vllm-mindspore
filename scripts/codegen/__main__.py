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

"""Main entry point for codegen."""

import constants as C
from json_utils import load_ops_def_from_json_files
from gen_aclnn_ops import AclnnOpsGenerator


def gen_aclnn_ops_launch(ops_defs):
    """Generate aclnn launch code"""
    generator = AclnnOpsGenerator()
    generator.generate(ops_defs)

if __name__ == "__main__":
    # Load operation definitions from json
    ops_defs_info = load_ops_def_from_json_files(C.JSON_FILES_DIR)
    # Generate aclnn ops launch code
    gen_aclnn_ops_launch(ops_defs_info)
    print("generate ops code success")
