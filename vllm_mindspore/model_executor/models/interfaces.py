# SPDX-License-Identifier: Apache-2.0

# Adapted from
# https://github.com/vllm-project/vllm/blob/v0.8.3/vllm/model_executor/models/interfaces.py
#
# Copyright 2025 Huawei Technologies Co., Ltd.
# Copyright 2025 The vLLM team.
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

from collections.abc import Iterable, MutableSequence
from typing import (TYPE_CHECKING, ClassVar, Literal, Optional, Protocol,
                    Union, overload, runtime_checkable)

from mindspore import Tensor
from typing_extensions import TypeIs

if TYPE_CHECKING:
    from vllm.attention import AttentionMetadata

MultiModalEmbeddings = Union[list[Tensor], Tensor, tuple[Tensor, ...]]


@runtime_checkable
class SupportsMultiModal(Protocol):
    """The interface required for all multi-modal models."""

    supports_multimodal: ClassVar[Literal[True]] = True
    """
    A flag that indicates this model supports multi-modal inputs.

    Note:
        There is no need to redefine this flag if this class is in the
        MRO of your model class.
    """

    def get_multimodal_embeddings(
            self, **kwargs: object) -> Optional[MultiModalEmbeddings]:
        """
            Returns multimodal embeddings generated from multimodal kwargs 
            to be merged with text embeddings.

            Note:
                The returned multimodal embeddings must be in the same order as
                the appearances of their corresponding multimodal data item in
                the input prompt.
            """
        ...

    # Only for models that support v0 chunked prefill
    # TODO(ywang96): Remove this overload once v0 is deprecated
    @overload
    def get_input_embeddings(
        self,
        input_ids: Tensor,
        multimodal_embeddings: Optional[MultiModalEmbeddings] = None,
        attn_metadata: Optional["AttentionMetadata"] = None,
    ) -> Tensor:
        ...

    @overload
    # type: ignore[misc]
    def get_input_embeddings(
        self,
        input_ids: Tensor,
        multimodal_embeddings: Optional[MultiModalEmbeddings] = None,
    ) -> Tensor:
        """
            Returns the input embeddings merged from the text embeddings from 
            input_ids and the multimodal embeddings generated from multimodal 
            kwargs.
            """
        ...


@runtime_checkable
class SupportsLoRA(Protocol):
    """The interface required for all models that support LoRA."""

    supports_lora: ClassVar[Literal[True]] = True
    """
    A flag that indicates this model supports LoRA.

    Note:
        There is no need to redefine this flag if this class is in the
        MRO of your model class.
    """

    packed_modules_mapping: ClassVar[dict[str, list[str]]]
    supported_lora_modules: ClassVar[list[str]]
    embedding_modules: ClassVar[dict[str, str]]
    embedding_padding_modules: ClassVar[list[str]]


@runtime_checkable
class _SupportsLoRAType(Protocol):
    supports_lora: Literal[True]

    packed_modules_mapping: dict[str, list[str]]
    supported_lora_modules: list[str]
    embedding_modules: dict[str, str]
    embedding_padding_modules: list[str]


@runtime_checkable
class MixtureOfExperts(Protocol):
    """
    Check if the model is a mixture of experts (MoE) model.
    """

    expert_weights: MutableSequence[Iterable[Tensor]]
    """
    Expert weights saved in this rank.

    The first dimension is the layer, and the second dimension is different
    parameters in the layer, e.g. up/down projection weights.
    """

    num_moe_layers: int
    """Number of MoE layers in this model."""

    num_expert_groups: int
    """Number of expert groups in this model."""

    num_logical_experts: int
    """Number of logical experts in this model."""

    num_physical_experts: int
    """Number of physical experts in this model."""

    num_local_physical_experts: int
    """Number of local physical experts in this model."""

    num_routed_experts: int
    """Number of routed experts in this model."""

    num_shared_experts: int
    """Number of shared experts in this model."""

    num_redundant_experts: int
    """Number of redundant experts in this model."""


def is_mixture_of_experts(model: object) -> TypeIs[MixtureOfExperts]:
    return isinstance(model, MixtureOfExperts)


@runtime_checkable
class SupportesMoeDpTp(Protocol):
    """
    The interface required for MoE models that MoE layer
    supports TP parallel under DP Context.
    """
    support_moe_dp_tp: ClassVar[Literal[True]] = True


def supports_moe_dp_tp(model: object) -> bool:
    return getattr(model, "support_moe_dp_tp", False)
