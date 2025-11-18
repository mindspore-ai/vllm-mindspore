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
"""Adaption for input processor."""

from vllm.inputs.registry import (BatchFeature, InputProcessingContext,
                                  Mapping, ProcessorMixin)

origin_call_hf_processor = InputProcessingContext.call_hf_processor


def call_hf_processor(
        self,
        hf_processor: ProcessorMixin,
        data: Mapping[str, object],
        kwargs: Mapping[str, object] = {},  # noqa: B006
) -> BatchFeature:
    """
    Call :code:`hf_processor` to get numpy tensors.
    """

    def _wrapper(func):

        def _inner(*args, **kwargs):
            kwargs["return_tensors"] = "np"
            return func(*args, **kwargs)

        return _inner

    wrapped_hf_processor = _wrapper(hf_processor)
    return origin_call_hf_processor(self, wrapped_hf_processor, data, kwargs)
