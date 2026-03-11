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

# pylint: disable=missing-module-docstring
# Import placed after env setup due to NPU library dependencies.
import os
try:
    import torch_npu
    torch_npu_path = os.path.dirname(torch_npu.__file__)
except ImportError:
    torch_npu_path = None
if torch_npu_path:
    lib_path = os.path.join(torch_npu_path, "lib")
    ld_lib_path = os.environ.get("LD_LIBRARY_PATH")
    os.environ["LD_LIBRARY_PATH"] = f"{lib_path}:{ld_lib_path}"

# pylint: disable=wrong-import-position
from ms_inferrt.torch.fx_backend import backend, register_dvm_op, get_dvm_payload
from ms_inferrt.torch.fx_mlir_backend import backend as fx_mlir_backend

__all__ = ['backend', 'fx_mlir_backend', 'register_dvm_op', 'get_dvm_payload']
