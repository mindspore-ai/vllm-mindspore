#!/usr/bin/env python3
# encoding: utf-8
# Copyright 2025 Huawei Technologies Co., Ltd
# Copyright 2024 The vLLM team.
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
# ============================================================================
from vllm.inputs.registry import ProcessorMixin, Mapping, BatchFeature, resolve_mm_processor_kwargs


def call_hf_processor(
    self,
    hf_processor: ProcessorMixin,
    data: Mapping[str, object],
    kwargs: Mapping[str, object] = {},
) -> BatchFeature:
    """
    Call :code:`hf_processor` on the prompt :code:`data`
    (text, image, audio...) with configurable options :code:`kwargs`.
    """
    assert callable(hf_processor)

    base_kwargs = self.model_config.mm_processor_kwargs
    if base_kwargs is None:
        base_kwargs = {}

    merged_kwargs = resolve_mm_processor_kwargs(
        base_kwargs,
        kwargs,
        hf_processor,
        requires_kw_only=False,
        allow_var_kwargs=True,
    )
    # replace call_hf_processor of vLLM for multi-model
    try:
        return hf_processor(**data, **merged_kwargs, return_tensors="np")
    except Exception as exc:
        msg = f"Failed to apply {type(hf_processor).__name__}"

        raise RuntimeError(msg) from exc
