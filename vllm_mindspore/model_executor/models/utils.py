# SPDX-License-Identifier: Apache-2.0

# Adapted from
# https://github.com/vllm-project/vllm/blob/v0.8.3/vllm/model_executor/models/utils.py
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

import itertools
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Optional, Union

import mindspore as ms
from mindspore import Parameter, Tensor, mint, ops
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors

from vllm_mindspore.model_executor.model_loader.weight_utils import (
    default_weight_loader)
from vllm_mindspore.multimodal.inputs import NestedTensors
from vllm_mindspore.utils import get_valid_dtype, is_310p

WeightsMapping = Mapping[str, Optional[str]]
"""If a key maps to a value of `None`, the corresponding weight is ignored."""

logger = init_logger(__name__)


def convert_pin(input_tensor):
    """Convert tensor to pinned memory if it's on CPU and not already pinned.
    
    Args:
        input_tensor: Input tensor to convert
        
    Returns:
        Tensor with pinned memory if applicable, otherwise original tensor
    """
    if not isinstance(input_tensor, Tensor):
        return input_tensor
    if input_tensor._ms_device == "CPU" and not input_tensor.is_pinned():
        input_pined = input_tensor.pin_memory()
        return input_pined
    return input_tensor


@dataclass
class WeightsMapper:
    """Maps the name of each weight if they match the following patterns."""

    orig_to_new_substr: WeightsMapping = field(default_factory=dict)
    orig_to_new_prefix: WeightsMapping = field(default_factory=dict)
    orig_to_new_suffix: WeightsMapping = field(default_factory=dict)

    def _map_name(self, key: str) -> Optional[str]:
        for substr, new_key in self.orig_to_new_substr.items():
            if substr in key:
                if new_key is None:
                    return None

                key = key.replace(substr, new_key, 1)

        for prefix, new_key in self.orig_to_new_prefix.items():
            if key.startswith(prefix):
                if new_key is None:
                    return None

                key = key.replace(prefix, new_key, 1)

        for suffix, new_key in self.orig_to_new_suffix.items():
            if key.endswith(suffix):
                if new_key is None:
                    return None

                key = new_key.join(key.rsplit(suffix, 1))

        return key

    def apply(
        self, weights: Iterable[tuple[str, ms.Tensor]]
    ) -> Iterable[tuple[str, ms.Tensor]]:
        return ((out_name, data) for name, data in weights
                if (out_name := self._map_name(name)) is not None)


enforce_eager = False


class PPMissingLayer(ms.nn.Cell):
    """
    A placeholder layer for missing layers in a pipeline parallel model.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()

    def construct(self, inputs):
        return inputs


def maybe_offload_to_cpu(module):
    # TODO: support
    return module


def maybe_prefix(prefix: str, name: str) -> str:
    """Add a prefix to a name if the prefix is non-empty.

    Args:
        prefix: The prefix to add. If empty, no prefix will be added.
        name: The name to potentially prefix.

    Returns:
        The string "prefix.name" if prefix was non-empty, otherwise just "name".
    """
    return name if not prefix else f"{prefix}.{name}"


def extract_layer_index(layer_name: str) -> int:
    """
    Extract the layer index from the module name.
    Examples:
    - "encoder.layers.0" -> 0
    - "encoder.layers.1.self_attn" -> 1
    - "2.self_attn" -> 2
    - "model.encoder.layers.0.sub.1" -> ValueError
    """
    subnames = layer_name.split(".")
    int_vals: list[int] = []
    for subname in subnames:
        try:
            int_vals.append(int(subname))
        except ValueError:
            continue
    assert len(int_vals) == 1, (f"layer name {layer_name} should"
                                " only contain one integer")
    return int_vals[0]


def make_layers(
    num_hidden_layers: int,
    layer_fn,
    prefix: str,
) -> tuple[int, int, ms.nn.CellList]:
    """Make a list of layers with the given layer function, taking
    pipeline parallelism into account.
    """
    from vllm.distributed.parallel_state import get_pp_group
    from vllm.distributed.utils import get_pp_indices

    start_layer, end_layer = get_pp_indices(num_hidden_layers,
                                            get_pp_group().rank_in_group,
                                            get_pp_group().world_size)
    modules = ms.nn.CellList([PPMissingLayer() for _ in range(start_layer)] + [
        maybe_offload_to_cpu(layer_fn(prefix=f"{prefix}.{idx}"))
        for idx in range(start_layer, end_layer)
    ] + [PPMissingLayer() for _ in range(end_layer, num_hidden_layers)])
    return start_layer, end_layer, modules


# NOTE: don't use lru_cache here because it can prevent garbage collection
_model_to_pp_missing_layer_names: dict[int, list[str]] = {}


def get_pp_missing_layer_names(model: ms.nn.Cell) -> list[str]:
    """Get the names of the missing layers in a pipeline parallel model."""
    model_id = id(model)
    if model_id in _model_to_pp_missing_layer_names:
        return _model_to_pp_missing_layer_names[model_id]

    missing_layer_names = []
    for name, module in model.cells_and_names():
        if isinstance(module, PPMissingLayer):
            # NOTE: the trailing dot is used to match the prefix of the layer.
            # without the dot, we could match a layer that is not missing,
            # e.g., 'encoder.layer.1' would match 'encoder.layer.11'
            missing_layer_names.append(name + '.')
    _model_to_pp_missing_layer_names[model_id] = missing_layer_names

    return missing_layer_names


def is_pp_missing_parameter(name: str, model: ms.nn.Cell) -> bool:
    """Check if a parameter is missing in a pipeline parallel model."""
    if isinstance(model, PPMissingLayer):
        return True

    return any(
        name.startswith(missing_layer_name)
        for missing_layer_name in get_pp_missing_layer_names(model))


def make_empty_intermediate_tensors_factory(keys: list[str], hidden_size: int):

    def make_empty_intermediate_tensors(
        batch_size: int,
        dtype,
        device,
    ) -> IntermediateTensors:
        dtype = get_valid_dtype(dtype)
        return IntermediateTensors({
            key:
            mint.zeros((batch_size, hidden_size), dtype=dtype)
            for key in keys
        })

    return make_empty_intermediate_tensors


########################### for multi model ###########################


def _flatten_embeddings(embeddings: NestedTensors) -> ms.Tensor:
    """
    Recursively flattens and concatenates NestedTensors on all but the last
    dimension.
    """

    if isinstance(embeddings, ms.Tensor):
        # Flatten all but the last dimension.
        return embeddings.flatten(0, -2)

    return ops.cat(tuple(_flatten_embeddings(t) for t in embeddings))


def _embedding_count_expression(embeddings: NestedTensors) -> str:
    """
    Constructs a debugging representation of the number of embeddings in the
    NestedTensors.
    """

    if isinstance(embeddings, ms.Tensor):
        return " x ".join([str(dim) for dim in embeddings.shape[:-1]])

    return " + ".join(
        _embedding_count_expression(inner) for inner in embeddings)


def _merge_multimodal_embeddings(
    inputs_embeds: ms.Tensor,
    is_multimodal: ms.Tensor,
    multimodal_embeddings: NestedTensors,
) -> ms.Tensor:
    """
    Merge ``multimodal_embeddings`` into ``inputs_embeds`` by overwriting the
    positions in ``inputs_embeds`` corresponding to placeholder tokens in
    ``input_ids``.

    Note:
        This updates ``inputs_embeds`` in place.
    """
    num_expected_tokens = is_multimodal.sum().item()
    assert isinstance(num_expected_tokens, int)

    flattened = _flatten_embeddings(multimodal_embeddings)
    if flattened.shape[0] != num_expected_tokens:
        expr = _embedding_count_expression(multimodal_embeddings)
        raise ValueError(
            f"Attempted to assign {expr} = {flattened.shape[0]} "
            f"multimodal tokens to {num_expected_tokens} placeholders")

    inputs_embeds[is_multimodal] = flattened
    return inputs_embeds


def merge_multimodal_embeddings(
    input_ids: ms.Tensor,
    inputs_embeds: ms.Tensor,
    multimodal_embeddings: NestedTensors,
    placeholder_token_id: Union[int, list[int]],
) -> ms.Tensor:
    """
    Merge ``multimodal_embeddings`` into ``inputs_embeds`` by overwriting the
    positions in ``inputs_embeds`` corresponding to placeholder tokens in
    ``input_ids``.
    
    ``placeholder_token_id`` can be a list of token ids (e.g, token ids 
    of img_start, img_break, and img_end tokens) when needed: This means 
    the order of these tokens in the ``input_ids`` MUST MATCH the order of 
    their embeddings in ``multimodal_embeddings`` since we need to 
    slice-merge instead of individually scattering.

    For example, if input_ids is "TTTTTSIIIBIIIBIIIETTT", where
    - T is text token
    - S is image start token
    - I is image embedding token
    - B is image break token
    - E is image end token.
    
    Then the image embeddings (that correspond to I's) from vision encoder 
    must be padded with embeddings of S, B, and E in the same order of 
    input_ids for a correct embedding merge.

    Note:
        This updates ``inputs_embeds`` in place.
    """
    if isinstance(placeholder_token_id, list):
        placeholder_token_id = ms.Tensor(placeholder_token_id)
        return _merge_multimodal_embeddings(
            inputs_embeds,
            ms.numpy.isin(input_ids, placeholder_token_id),
            multimodal_embeddings,
        )

    return _merge_multimodal_embeddings(
        inputs_embeds,
        (input_ids == placeholder_token_id),
        multimodal_embeddings,
    )


def is_use_ringmla(vllm_config, mf_config=None):
    """
    Determine whether MLA model uses RingMLA
    """
    try:
        import ms_custom_ops  # noqa: F401
    except ModuleNotFoundError:
        # environment need install ms_custom_ops package
        return False
    if is_310p():
        return False

    fa3_quant = vllm_config.cache_config.cache_dtype == "int8"
    if vllm_config.speculative_config is not None and not fa3_quant:
        # For speculative inference, ringmla may reduce the acceptance rate.
        # So that ringmla is enabled only if fa3_quant is enabled, and disabled
        # when the base model is w8a8-quantized.
        return False

    use_ringmla = (vllm_config.model_config.use_mla
                   and vllm_config.model_config.quantization is not None
                   and vllm_config.parallel_config.tensor_parallel_size < 16)
    return use_ringmla


class AutoWeightsLoaderMS:
    """
    Helper class to load weights into a [`nn.Cell`][]. It is able
    to automatically detect child modules and parameters while iterating over
    the weights only once.

    The weight loading logic for individual modules can be overridden
    by defining a ``load_weights`` method.

    Similarly, the weight loading logic for individual parameters can be
    overridden by defining a ``weight_loader`` method.

    Detailed weight loading information can be viewed by setting the
    environment variable ``VLLM_LOGGING_LEVEL=DEBUG``.
    """

    # Models trained using early version ColossalAI
    # may include these tensors in checkpoint. Skip them.
    ROTARY_EMBEDS_UNUSED_WEIGHTS = [
        "rotary_emb.inv_freq",
        "rotary_emb.cos_cached",
        "rotary_emb.sin_cached",
    ]

    def __init__(
        self,
        module,
        *,
        skip_prefixes: Optional[list[str]] = None,
        skip_substrs: Optional[list[str]] = None,
        ignore_unexpected_prefixes: Optional[list[str]] = None,
    ) -> None:
        super().__init__()

        self.module = module
        self.skip_prefixes = skip_prefixes or []
        self.skip_substrs = skip_substrs or []
        self.ignore_unexpected_prefixes = ignore_unexpected_prefixes or []
        # update default skip_substrs
        self.skip_substrs += self.ROTARY_EMBEDS_UNUSED_WEIGHTS

    def _can_skip(self, qualname: str) -> bool:
        return (any(qualname.startswith(p) for p in self.skip_prefixes)
                or any(substr in qualname for substr in self.skip_substrs))

    def _can_ignore_unexpected(self, qualname: str) -> bool:
        return any(
            qualname.startswith(p) for p in self.ignore_unexpected_prefixes)

    def _groupby_prefix(
        self,
        weights: Iterable[tuple[str, Tensor]],
    ) -> Iterable[tuple[str, Iterable[tuple[str, Tensor]]]]:
        modules_name = self.module.get_modules_dict().keys()

        # Define key func: find matching prefix (module name) in name
        def get_prefix(name):
            for prefix in modules_name:
                if prefix == "":
                    if "." not in name:
                        return ""
                elif name.startswith(prefix + ".") or name == prefix:
                    # Ensure prefix does not end with '.'
                    return prefix.rstrip('.')
            return ""

        # Group by prefix
        for prefix, group in itertools.groupby(weights,
                                               key=lambda x: get_prefix(x[0])):
            yield (prefix, ((parts, weights_data)
                            for parts, weights_data in group))

    def _load_module(
        self,
        base_prefix: str,
        module,
        weights: Iterable[tuple[str, Tensor]],
        param_dict: dict[str, Parameter],
    ) -> Iterable[str]:
        msg = ("There is no module or parameter named %s "
               "in %s")

        modules_dict = module.get_modules_dict()
        for child_prefix, child_weights in self._groupby_prefix(weights):
            if self._can_skip(child_prefix + '.'):
                logger.debug("Skipping module %s", child_prefix)
                continue

            if self._can_ignore_unexpected(child_prefix + "."):
                logger.debug("Ignoring missing %s", child_prefix)
                continue

            module = modules_dict.get(child_prefix, None)
            if module is None:
                raise ValueError(f'The prefix "{child_prefix}" of weight '
                                 'is not found in model definition')

            if isinstance(module, PPMissingLayer):
                continue

            # For submodule, e.g., Language_model, visual model.
            if hasattr(module, "load_weights"):
                yield from modules_dict[child_prefix].load_weights(
                    child_weights, param_dict)
            # For independent cell, which is define under Native Model.
            # Or parameters rights under the Native Model
            else:
                memo = set()
                for name, weight in child_weights:
                    if self._can_skip(name):
                        logger.debug("Skipping param %s", name)
                        continue
                    if self._can_ignore_unexpected(name):
                        logger.debug("Ignoring missing %s", name)
                        continue
                    param = param_dict.get(name)
                    if param is None:
                        raise ValueError(
                            msg.format(name,
                                       type(self.module).__name__))
                    assert isinstance(
                        param, Parameter), f"param {name} is not a Parameter"
                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                    weight_loader(param, weight)
                    logger.debug("Loaded weight %s with shape %s", name,
                                 param.shape)
                    memo.add(name)
                yield from memo

    def load_weights(
        self,
        weights: Iterable[tuple[str, Tensor]],
        *,
        mapper: Optional[WeightsMapper] = None,
    ) -> set[str]:
        if mapper is not None:
            weights = mapper.apply(weights)
        # filter out weights with first-prefix/substr to skip in name
        weights = ((name, weight) for name, weight in weights
                   if not self._can_skip(name))

        param_dict = self.module.get_params_dict()
        autoloaded_weights = set(
            self._load_module("", self.module, weights, param_dict))
        return autoloaded_weights
