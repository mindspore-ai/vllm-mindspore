# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2024 The Qwen team.
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
"""Inference-only Qwen2-Audio model compatible with HuggingFace weights."""

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, Optional, TypedDict, Union

from transformers.models.qwen2_audio import (
    Qwen2AudioConfig,
    Qwen2AudioEncoder,
    Qwen2AudioProcessor,
)

from transformers.models.whisper import WhisperFeatureExtractor

from vllm.multimodal.processing import (
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)

from mindspore import Tensor

# # === Audio Inputs === #
class Qwen2AudioInputs(TypedDict):
    input_features: Tensor
    """Shape: `(num_audios, num_mel_bins, 3000)`"""

    feature_attention_mask: Tensor
    """Shape: `(num_audios, 3000)`"""


# === Audio Encoder === #

# From Qwen2AudioEncoder._get_feat_extract_output_lengths
def _get_feat_extract_output_lengths(input_lengths: Tensor):
    feat_lengths = (input_lengths - 1) // 2 + 1
    output_lengths = (feat_lengths - 2) // 2 + 1
    return feat_lengths, output_lengths


class Qwen2AudioProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self):
        return self.ctx.get_hf_config(Qwen2AudioConfig)

    def get_hf_processor(self, **kwargs: object) -> Qwen2AudioProcessor:
        return self.ctx.get_hf_processor(Qwen2AudioProcessor, **kwargs)

    def get_feature_extractor(self, **kwargs: object) -> WhisperFeatureExtractor:
        hf_processor = self.get_hf_processor(**kwargs)
        feature_extractor = hf_processor.feature_extractor  # type: ignore
        assert isinstance(feature_extractor, WhisperFeatureExtractor)
        return feature_extractor

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"audio": None}


# class Qwen2AudioFeatureInputs(TensorSchema):
#     """
#     Dimensions:
#         - na: Number of audios
#         - nmb: Number of mel bins
#     """

#     type: Literal["audio_features"]
#     input_features: Annotated[
#         Tensor | list[Tensor],
#         TensorShape("na", "nmb", 3000),
#     ]

#     feature_attention_mask: Annotated[
#         Tensor,
#         TensorShape("na", 3000),
#     ]
