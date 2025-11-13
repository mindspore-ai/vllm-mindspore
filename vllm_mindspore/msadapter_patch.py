# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Patching msadapter functions to use view(dtype)"""
import sys
from types import ModuleType

import msadapter


def patch_msadapter():
    # torch.Tag.needs_fixed_stride_order
    if not hasattr(msadapter, 'Tag'):

        class Tag:

            @property
            def needs_fixed_stride_order(self):
                return None

        msadapter.Tag = Tag

    # torch.ops.vllm.dequant_mxfp4
    if not hasattr(msadapter.ops, 'vllm'):

        class VLLMOps:
            pass

        msadapter.ops.vllm = VLLMOps()
        msadapter.ops.vllm.dequant_mxfp4 = None
        msadapter.ops.vllm.quant_dequant_mxfp4 = None

    # torch.utils.cpp_extension.load_inline
    if not hasattr(msadapter.utils, 'cpp_extension'):
        cpp_extension_module = ModuleType('cpp_extension')
        msadapter.utils.cpp_extension = cpp_extension_module
        sys.modules['msadapter.utils.cpp_extension'] = cpp_extension_module
        msadapter.utils.cpp_extension.load_inline = None

    # torch.autograd.profiler.record_function
    if not hasattr(msadapter.autograd, 'profiler'):
        profiler_module = ModuleType('profiler')
        msadapter.autograd.profiler = profiler_module
        sys.modules['msadapter.autograd.profiler'] = profiler_module
        msadapter.autograd.profiler.record_function = None

    # torch._subclasses.fake_tensor.FakeTensorMode
    # torch._subclasses.fake_tensor.unset_fake_temporarily
    if not hasattr(msadapter, '_subclasses'):
        _subclasses_module = ModuleType('_subclasses')
        msadapter._subclasses = _subclasses_module
        sys.modules['msadapter._subclasses'] = _subclasses_module
        fake_tensor_module = ModuleType('fake_tensor')
        msadapter._subclasses.fake_tensor = fake_tensor_module
        sys.modules['msadapter._subclasses.fake_tensor'] = fake_tensor_module
        msadapter._subclasses.fake_tensor.FakeTensorMode = None
        msadapter._subclasses.fake_tensor.unset_fake_temporarily = None

    # torch.cuda.CUDAGraph
    if not hasattr(msadapter.cuda, 'CUDAGraph'):
        msadapter.cuda.CUDAGraph = None

    # torch.tensor.to
    PT_MS_DEVICE_TYPE_MAP = {
        'cuda': 'Ascend',
        'npu': 'Ascend',
        'Ascend': 'Ascend',
        'cpu': 'CPU'
    }

    def _device_to_ms(device):
        if isinstance(device, str):
            torch_device_type = msadapter.device(device).type
            return PT_MS_DEVICE_TYPE_MAP[torch_device_type]
        elif isinstance(device, msadapter.device):
            if device.type == "meta":
                return None
            return PT_MS_DEVICE_TYPE_MAP[device.type]
        else:
            raise TypeError(
                f"device must be str/torch.device, not {type(device).__name__}"
            )

    msadapter._tensor._device_to_ms = _device_to_ms
