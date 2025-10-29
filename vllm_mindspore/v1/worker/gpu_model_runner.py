# SPDX-License-Identifier: Apache-2.0

# Adapted from
# https://github.com/vllm-project/vllm/blob/v0.8.3/vllm/v1/worker/gpu_model_runner.py
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

import traceback
from typing import Any, Optional

import mindspore as ms
import numpy as np
import torch
from mindspore import Generator as msGenerator
from mindspore import Tensor, mint, mutable
from vllm.attention import AttentionType
from vllm.config import get_layers_from_vllm_config
from vllm.logger import init_logger
from vllm.sampling_params import SamplingType
from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheConfig,
                                        KVCacheSpec, SlidingWindowSpec)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.utils import bind_kv_cache
from vllm.v1.worker.gpu_input_batch import CachedRequestState
from vllm.v1.worker.utils import initialize_kv_cache_for_kv_sharing

from vllm_mindspore.model_executor.layers.rotary_embedding import (
    InferMRotaryEmbedding as MRotaryEmbedding)
from vllm_mindspore.model_executor.models.model_base import AttentionWrapper
from vllm_mindspore.model_executor.models.utils import is_use_ringmla
from vllm_mindspore.utils import (create_kv_cache, get_dtype_size,
                                  get_valid_dtype, is_310p)
from vllm_mindspore.v1.kv_cache_interface import MLAQuantFullAttentionSpec

logger = init_logger(__name__)


def _prepare_inputs(
    self,
    scheduler_output,
) -> tuple[dict[str, Any], Tensor, Optional[SpecDecodeMetadata]]:
    total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
    assert total_num_scheduled_tokens > 0
    num_reqs = self.input_batch.num_reqs
    assert num_reqs > 0

    # OPTIMIZATION: Start copying the block table first.
    # This way, we can overlap the copy with the following CPU operations.
    self.input_batch.block_table.commit(num_reqs)

    # Get the number of scheduled tokens for each request.
    req_ids = self.input_batch.req_ids
    tokens = [scheduler_output.num_scheduled_tokens[i] for i in req_ids]
    num_scheduled_tokens = np.array(tokens, dtype=np.int32)
    max_num_scheduled_tokens = max(tokens)

    # Get request indices.
    # E.g., [2, 5, 3] -> [0, 0, 1, 1, 1, 1, 1, 2, 2, 2]
    req_indices = np.repeat(self.arange_np[:num_reqs], num_scheduled_tokens)
    """
    cu_num_tokens: [2, 5, 3] -> [2, 7, 10]
    arange: [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
    """
    cu_num_tokens, arange = self._get_cumsum_and_arange(num_scheduled_tokens)

    # Get positions.
    positions_np = self.positions_np[:total_num_scheduled_tokens]
    np.add(self.input_batch.num_computed_tokens_cpu[req_indices],
           arange,
           out=positions_np)

    # Calculate M-RoPE positions.
    # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
    if self.uses_mrope:
        self._calc_mrope_positions(scheduler_output)

    # Get token indices.
    # E.g., [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
    # -> [0, 1, M, M + 1, M + 2, M + 3, M + 4, 2 * M, 2 * M + 1, 2 * M + 2]
    # where M is the max_model_len.
    token_indices = (positions_np +
                     req_indices * self.input_batch.token_ids_cpu.shape[1])

    # vllm-mindspore begin
    self.input_ids[:total_num_scheduled_tokens] = ms.from_numpy(
        np.take(self.input_batch.token_ids_cpu.ravel(), token_indices, 0))
    # vllm-mindspore end

    # Calculate the slot mapping for each KV cache group.
    for kv_cache_group_id, kv_cache_group_spec in enumerate(
            self.kv_cache_config.kv_cache_groups):
        block_size = kv_cache_group_spec.kv_cache_spec.block_size
        block_table = self.input_batch.block_table[kv_cache_group_id]
        # E.g., [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        # -> [0, 0, K, K, K + 1, K + 1, K + 2, 2 * K, 2 * K, 2 * K + 1]
        # where K is the max_num_blocks_per_req and the block size is 2.
        # NOTE(woosuk): We can't simply use `token_indices // block_size`
        # here because M (max_model_len) is not necessarily divisible by
        # block_size.
        block_table_indices = (
            req_indices * block_table.max_num_blocks_per_req +
            positions_np // block_size)
        # vllm-mindspore begin
        block_numbers = block_table.block_table_np.ravel()[block_table_indices]
        # vllm-mindspore end
        block_offsets = positions_np % block_size
        np.add(block_numbers * block_size,
               block_offsets,
               out=block_table.slot_mapping_np[:total_num_scheduled_tokens])

    # # Prepare the attention metadata.
    self.query_start_loc_np[0] = 0
    self.query_start_loc_np[1:num_reqs + 1] = cu_num_tokens

    self.seq_lens_np[:num_reqs] = (
        self.input_batch.num_computed_tokens_cpu[:num_reqs] +
        num_scheduled_tokens)

    if self.uses_mrope:
        # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
        self.mrope_positions[:, :total_num_scheduled_tokens] = \
            self.mrope_positions_cpu[:, :total_num_scheduled_tokens]
    else:
        # Common case (1D positions)
        # vllm-mindspore begin
        self.positions[:total_num_scheduled_tokens] = ms.from_numpy(
            positions_np)
        # vllm-mindspore end

    self.query_start_loc[:num_reqs + 1] = self.query_start_loc_cpu[:num_reqs +
                                                                   1]
    self.seq_lens[:num_reqs] = self.seq_lens_cpu[:num_reqs]

    # Fill unused with -1. Needed for reshape_and_cache
    self.seq_lens[num_reqs:].fill_(0)
    # Note: pad query_start_loc to be non-decreasing, as kernels
    # like FlashAttention requires that
    self.query_start_loc[num_reqs + 1:].fill_(
        self.query_start_loc_cpu[num_reqs].item())

    # vllm-mindspore begin
    query_start_loc = ms.from_numpy(self.query_start_loc_np[:num_reqs + 1])

    attn_metadata = {}
    # vllm-mindspore end

    # Prepare the attention metadata for each KV cache group and make layers
    # in the same group share the same metadata.
    for kv_cache_group_id, kv_cache_group_spec in enumerate(
            self.kv_cache_config.kv_cache_groups):

        # Prepare for cascade attention if enabled & beneficial.
        common_prefix_len = 0
        if self.cascade_attn_enabled:
            common_prefix_len = self._compute_cascade_attn_prefix_len(
                num_scheduled_tokens,
                scheduler_output.num_common_prefix_blocks[kv_cache_group_id],
                kv_cache_group_spec.kv_cache_spec,
                self.attn_metadata_builders[kv_cache_group_id],
            )

        attn_metadata_i = (
            self.attn_metadata_builders[kv_cache_group_id].build(
                num_reqs=num_reqs,
                num_actual_tokens=total_num_scheduled_tokens,
                max_query_len=max_num_scheduled_tokens,
                common_prefix_len=common_prefix_len,
            ))
        for layer_name in kv_cache_group_spec.layer_names:
            attn_metadata[layer_name] = attn_metadata_i

    use_spec_decode = len(scheduler_output.scheduled_spec_decode_tokens) > 0
    if not use_spec_decode:
        # NOTE(woosuk): Due to chunked prefills, the batch may contain
        # partial requests. While we should not sample any token
        # from these partial requests, we do so for simplicity.
        # We will ignore the sampled tokens from the partial requests.
        # TODO: Support prompt logprobs.
        logits_indices = query_start_loc[1:] - 1
        spec_decode_metadata = None
    else:
        # Get the number of draft tokens for each request.
        # Iterate over the dictionary rather than all requests since not all
        # requests have draft tokens.
        num_draft_tokens = np.zeros(num_reqs, dtype=np.int32)
        for req_id, draft_token_ids in (
                scheduler_output.scheduled_spec_decode_tokens.items()):
            req_idx = self.input_batch.req_id_to_index[req_id]
            num_draft_tokens[req_idx] = len(draft_token_ids)

        spec_decode_metadata = self._calc_spec_decode_metadata(
            num_draft_tokens, cu_num_tokens)
        logits_indices = spec_decode_metadata.logits_indices

    # Hot-Swap lora model
    if self.lora_config:
        self.set_active_loras(self.input_batch, num_scheduled_tokens)

    return attn_metadata, logits_indices, spec_decode_metadata


def create_block(shape, dtype, name=None, device=None):
    blocks = mint.empty(shape, dtype=dtype, device=device)
    return blocks


def _allocate_nz_kv_cache_tensors(self, kv_cache_config):
    """
    Initializes and reshape the KV cache buffer with the correct size. 
    The buffer needs to be convert to nz format for 310p.

    Args:
        kv_cache_config: The KV cache config 
    Returns:
        dict[str, Tensor]: A map between layer names to their 
        corresponding memory buffer for KV cache.
    """
    kv_caches: dict[str, tuple] = {}

    layer_to_group_info = {
        layer_name: (i, group.kv_cache_spec)
        for i, group in enumerate(kv_cache_config.kv_cache_groups)
        for layer_name in group.layer_names
    }
    # Determine whether deepseek use mla op
    use_ringmla = is_use_ringmla(self.vllm_config)
    if use_ringmla:
        logger.error("For 310p, mla kv cache not supported")
        raise NotImplementedError

    for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
        if not kv_cache_tensor.shared_by:
            continue

        rep_layer_name = kv_cache_tensor.shared_by[0]
        group_idx, kv_cache_spec = layer_to_group_info[rep_layer_name]
        if not isinstance(kv_cache_spec, FullAttentionSpec):
            raise NotImplementedError

        attn_backend = self.attn_backends[group_idx]
        target_dtype = get_valid_dtype(kv_cache_spec.dtype)

        num_blocks = kv_cache_tensor.size // kv_cache_spec.page_size_bytes
        kv_cache_shape = attn_backend.get_kv_cache_shape(
            num_blocks, kv_cache_spec.block_size, kv_cache_spec.num_kv_heads,
            kv_cache_spec.head_size)

        reshaped_layer_tensors = []
        coef = 1 if kv_cache_spec.use_mla else 2
        for _ in range(coef):
            reshaped_layer_tensors.append(
                create_kv_cache(kv_cache_shape[1:], target_dtype))

        final_kv_tuple = mutable(tuple(reshaped_layer_tensors))
        for layer_name in kv_cache_tensor.shared_by:
            kv_caches[layer_name] = final_kv_tuple

    all_layers = set(layer_to_group_info.keys())
    if all_layers != set(kv_caches.keys()):
        raise RuntimeError("Some layers were not initialized")

    return kv_caches


def _allocate_nz_kv_cache_tensors_fa3(self, kv_cache_config):
    """
    Initializes and reshape the KV cache buffer with the correct size. 
    The buffer needs to be convert to nz format for fa3.
    Offloading kv_cache memory per layer and combine allocate and reshape
    kv cache together without constructing raw tensors

    Args:
        kv_cache_config: The KV cache config 
    Returns:
        dict[str, Tensor]: A map between layer names to their 
        corresponding memory buffer for KV cache.
    """
    kv_caches: dict[str, tuple] = {}

    layer_to_group_info = {
        layer_name: (i, group.kv_cache_spec)
        for i, group in enumerate(kv_cache_config.kv_cache_groups)
        for layer_name in group.layer_names
    }
    # fa3 quant layer target_dtype is int8
    # no fa3 quant layer target_dtype is bfloat16
    fa3_quant = self.vllm_config.quant_config.fa3_quant \
                    if self.vllm_config.quant_config else False
    fa3_quant_layer = self.vllm_config.quant_config.fa3_quant_layer \
                        if self.vllm_config.quant_config else set()
    kv_lora_rank = getattr(self.vllm_config.model_config.hf_text_config,
                           'kv_lora_rank', 0)
    qk_rope_head_dim = getattr(self.vllm_config.model_config.hf_text_config,
                               'qk_rope_head_dim', 0)

    for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
        if not kv_cache_tensor.shared_by:
            continue
        rep_layer_name = kv_cache_tensor.shared_by[0]
        _, kv_cache_spec = layer_to_group_info[rep_layer_name]

        is_fa3_quant_layer = fa3_quant and int(rep_layer_name) \
            in fa3_quant_layer
        num_blocks = kv_cache_tensor.size // kv_cache_spec.get_page_size(
            is_fa3_quant_layer)
        block_size = kv_cache_spec.block_size

        kv_cache_layer = []
        """
        for fa3_quant_layer, k_cache is int8, v_cache is bfloat16
        for not_fa3_quant_layer, k_cache and v_cache are bfloat16
        k_cache shape:
            [num_block, block_size, 1(head_dim), 512(kv_lora_rank)]
        v_cache shape:
            [num_block, block_size, 1(head_dim), 64(qk_rope_head_dim)]
        and target_dtype is int8
        """
        k_dtype = ms.int8 if is_fa3_quant_layer else \
                  self.vllm_config.model_config.dtype
        v_dtype = self.vllm_config.model_config.dtype
        head_dim = 1  # head_dim usually = 1
        k_shape = (num_blocks, block_size, head_dim, kv_lora_rank)
        v_shape = (num_blocks, block_size, head_dim, qk_rope_head_dim)

        kv_cache_layer.extend([
            create_kv_cache(k_shape, k_dtype, fa3_quant),
            create_kv_cache(v_shape, v_dtype, fa3_quant)
        ])
        final_kv_tuple = mutable(tuple(kv_cache_layer))
        for layer_name in kv_cache_tensor.shared_by:
            kv_caches[layer_name] = final_kv_tuple

        ms.runtime.empty_cache()

    all_layers = set(layer_to_group_info.keys())
    if all_layers != set(kv_caches.keys()):
        raise RuntimeError("Some layers were not initialized")

    return kv_caches


def _allocate_kv_cache_tensors(self, kv_cache_config):
    """
    Initializes the KV cache buffer with the correct size. The buffer needs
    to be reshaped to the desired shape before being used by the models.

    Args:
        kv_cache_config: The KV cache config 
    Returns:
        dict[str, Tensor]: A map between layer names to their 
        corresponding memory buffer for KV cache.
    """
    kv_cache_spec = kv_cache_config.kv_cache_groups[0].kv_cache_spec
    use_mla = kv_cache_spec.use_mla
    dtype = kv_cache_spec.dtype
    coef = 1 if use_mla else 2
    # Determine whether deepseek use mla op
    use_ringmla = is_use_ringmla(self.vllm_config)
    kv_lora_rank = getattr(self.vllm_config.model_config.hf_text_config,
                           'kv_lora_rank', 0)
    qk_rope_head_dim = getattr(self.vllm_config.model_config.hf_text_config,
                               'qk_rope_head_dim', 0)

    kv_cache_raw_tensors: dict[str, Tensor] = {}
    target_dtype = get_valid_dtype(dtype)
    dtype_size = get_dtype_size(target_dtype)
    for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
        assert len(kv_cache_tensor.shared_by) == 1
        raw_tensors = []
        raw_tensor_shape = kv_cache_tensor.size // dtype_size // coef
        for i in range(coef):
            """
            Formulas for calculating each parameter:
            1. page_size = coef * self.block_size * self.num_kv_heads *
               self.head_size * get_dtype_size(self.dtype)
            2. num_blocks = kv_cache_tensors.size / page_size
            3. kv_cache_tensors.size = num_blocks * (coef *
               self.block_size * self.num_kv_heads * self.head_size *
               get_dtype_size(self.dtype))
            4. kv cache shape: num_blocks, block_size, num_kv_heads, head_size
            """
            if not use_ringmla:
                raw_tensors.extend(
                    [mint.zeros(raw_tensor_shape, dtype=target_dtype)])
            else:
                raw_tensors.extend([
                    mint.zeros(int(raw_tensor_shape * kv_lora_rank /
                                   (kv_lora_rank + qk_rope_head_dim)),
                               dtype=target_dtype),
                    # deepseek mla op need key cache and rope cache
                    mint.zeros(int(raw_tensor_shape * qk_rope_head_dim /
                                   (kv_lora_rank + qk_rope_head_dim)),
                               dtype=target_dtype)
                ])
        for layer_name in kv_cache_tensor.shared_by:
            kv_cache_raw_tensors[layer_name] = tuple(raw_tensors)

    layer_names = set()
    for group in kv_cache_config.kv_cache_groups:
        layer_names.update(group.layer_names)
    assert layer_names == set(kv_cache_raw_tensors.keys()
                              ), "Some layers are not correctly initialized"
    return kv_cache_raw_tensors


def _reshape_kv_cache_tensors(
    self,
    kv_cache_config,
    kv_cache_raw_tensors,
):
    """
    Reshape the KV cache tensors to the desired shape and dtype.

    Args:
        kv_cache_config: The KV cache config 
        kv_cache_raw_tensors: The KV cache buffer of each layer, with 
        correct size but uninitialized shape.
    Returns:
        Dict[str, Tensor]: A map between layer names to their 
        corresponding memory buffer for KV cache.
    """
    # Determine whether deepseek use mla op
    use_ringmla = is_use_ringmla(self.vllm_config)
    kv_lora_rank = getattr(self.vllm_config.model_config.hf_text_config,
                           'kv_lora_rank', 0)
    qk_rope_head_dim = getattr(self.vllm_config.model_config.hf_text_config,
                               'qk_rope_head_dim', 0)
    kv_caches: dict[str, tuple] = {}
    for i, kv_cache_group_spec in enumerate(kv_cache_config.kv_cache_groups):
        kv_cache_spec = kv_cache_group_spec.kv_cache_spec
        coef = 1 if kv_cache_spec.use_mla else 2
        for layer_name in kv_cache_group_spec.layer_names:
            raw_tensor = kv_cache_raw_tensors[layer_name]
            target_dtype = get_valid_dtype(kv_cache_spec.dtype)
            dtype_size = get_dtype_size(target_dtype)
            num_blocks = \
                (raw_tensor[0].numel() if not use_ringmla else \
                (raw_tensor[0].numel() + raw_tensor[1].numel())) * \
                coef * dtype_size // kv_cache_spec.page_size_bytes
            if isinstance(kv_cache_spec, FullAttentionSpec):
                kv_cache_shape = self.attn_backends[i].get_kv_cache_shape(
                    num_blocks, kv_cache_spec.block_size,
                    kv_cache_spec.num_kv_heads, kv_cache_spec.head_size)
                try:
                    kv_cache_stride_order = self.attn_backends[
                        i].get_kv_cache_stride_order()
                    assert len(kv_cache_stride_order) == len(kv_cache_shape)
                except (AttributeError, NotImplementedError):
                    kv_cache_stride_order = tuple(range(len(kv_cache_shape)))
                # The allocation respects the backend-defined stride order
                # to ensure the semantic remains consistent for each
                # backend. We first obtain the generic kv cache shape and
                # then permute it according to the stride order which could
                # result in a non-contiguous tensor.
                kv_cache_shape = tuple(kv_cache_shape[i]
                                       for i in kv_cache_stride_order)
                # Maintain original KV shape view.
                inv_order = [
                    kv_cache_stride_order.index(i) - 1
                    for i in range(len(kv_cache_stride_order))
                ]
                kv_cache_layer = []
                for idx, kv_cache_raw_tensor in enumerate(
                        kv_cache_raw_tensors[layer_name]):
                    if use_ringmla:
                        # deepseek mla op need key cache and rope cache
                        cache_shape = [
                            *(kv_cache_shape[1:-1]),
                            kv_lora_rank if idx == 0 else qk_rope_head_dim
                        ]
                        cache_block = kv_cache_raw_tensor.view(
                            cache_shape).permute(*inv_order[1:])
                    else:
                        cache_block = kv_cache_raw_tensor.view(
                            kv_cache_shape[1:]).permute(*inv_order[1:])
                    kv_cache_layer.append(cache_block)
                kv_caches[layer_name] = mutable(tuple(kv_cache_layer))
            else:
                raise NotImplementedError
    return kv_caches


def initialize_kv_cache_tensors(
        self, kv_cache_config: KVCacheConfig) -> dict[str, torch.Tensor]:
    """
    Initialize the memory buffer for KV cache.

    Args:
        kv_cache_config: The KV cache config
    Returns:
        Dict[str, torch.Tensor]: A map between layer names to their 
        corresponding memory buffer for KV cache.
    """
    if is_310p():
        kv_caches = _allocate_nz_kv_cache_tensors(self, kv_cache_config)
    elif getattr(getattr(self.vllm_config, "quant_config", None), \
                    "fa3_quant", False):
        kv_caches = _allocate_nz_kv_cache_tensors_fa3(self, kv_cache_config)
    else:
        # Initialize the memory buffer for KV cache
        kv_cache_raw_tensors = self._allocate_kv_cache_tensors(kv_cache_config)
        # Change the memory buffer to the desired shape
        kv_caches = self._reshape_kv_cache_tensors(kv_cache_config,
                                                   kv_cache_raw_tensors)

    # Setup `kv_cache_config` and `kv_caches` for models
    # with cross-layer KV sharing
    if self.shared_kv_cache_layers:
        initialize_kv_cache_for_kv_sharing(
            self.shared_kv_cache_layers,
            kv_cache_config.kv_cache_groups,
            kv_caches,
        )

    bind_kv_cache(kv_caches,
                  self.vllm_config.compilation_config.static_forward_context,
                  self.kv_caches)
    return kv_caches


def _update_states(self, scheduler_output) -> None:
    """Update the cached states and the persistent batch with the scheduler
    output.

    The updated states are used by the `_prepare_inputs` function to create
    the input GPU tensors for the model.

    The SamplingMetadata is updated and copied to the GPU if there is a
    new/resumed/paused/finished request in the batch.
    """
    # Remove finished requests from the cached states.
    for req_id in scheduler_output.finished_req_ids:
        self.requests.pop(req_id, None)
        self.encoder_cache.pop(req_id, None)
    # Remove the finished requests from the persistent batch.
    # NOTE(woosuk): There could be an edge case where finished_req_ids and
    # scheduled_req_ids overlap. This happens when a request is aborted and
    # then resubmitted with the same ID. In this case, we treat them as two
    # distinct requests - clearing the cached states for the first request
    # and handling the second as a new request.
    removed_req_indices: list[int] = []
    for req_id in scheduler_output.finished_req_ids:
        req_index = self.input_batch.remove_request(req_id)
        if req_index is not None:
            removed_req_indices.append(req_index)

    # Free the cached encoder outputs.
    for req_id, input_id in scheduler_output.free_encoder_input_ids:
        encoder_outputs = self.encoder_cache.get(req_id)
        if encoder_outputs is not None:
            encoder_outputs.pop(input_id, None)
            if not encoder_outputs:
                self.encoder_cache.pop(req_id, None)

    # Remove the unscheduled requests from the persistent batch.
    # NOTE(woosuk): The unscheduled requests are either preempted requests
    # or running requests that are not scheduled in this step. We remove
    # them from the persistent batch but keep their cached states since
    # they will be scheduled again sometime in the future.
    scheduled_req_ids = scheduler_output.num_scheduled_tokens.keys()
    cached_req_ids = self.input_batch.req_id_to_index.keys()
    unscheduled_req_ids = cached_req_ids - scheduled_req_ids
    # NOTE(woosuk): The persistent batch optimization assumes that
    # consecutive batches contain mostly the same requests. If batches
    # have low request overlap (e.g., alternating between two distinct
    # sets of requests), this optimization becomes very inefficient.
    for req_id in unscheduled_req_ids:
        req_index = self.input_batch.remove_request(req_id)
        assert req_index is not None
        removed_req_indices.append(req_index)

    req_ids_to_add: list[str] = []
    # Add new requests to the cached states.
    for new_req_data in scheduler_output.scheduled_new_reqs:
        req_id = new_req_data.req_id
        sampling_params = new_req_data.sampling_params
        if sampling_params.sampling_type == SamplingType.RANDOM_SEED:
            generator = msGenerator()
            generator.manual_seed(sampling_params.seed)
        else:
            generator = None

        self.requests[req_id] = CachedRequestState(
            req_id=req_id,
            prompt_token_ids=new_req_data.prompt_token_ids,
            mm_inputs=new_req_data.mm_inputs,
            mm_positions=new_req_data.mm_positions,
            sampling_params=sampling_params,
            generator=generator,
            block_ids=new_req_data.block_ids,
            num_computed_tokens=new_req_data.num_computed_tokens,
            output_token_ids=[],
            lora_request=new_req_data.lora_request,
        )

        # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
        if self.uses_mrope:
            image_grid_thw = []
            video_grid_thw = []
            second_per_grid_ts = []
            for mm_input in self.requests[req_id].mm_inputs:
                if mm_input.get("image_grid_thw") is not None:
                    image_grid_thw.extend(mm_input["image_grid_thw"].tolist())
                if mm_input.get("video_grid_thw") is not None:
                    video_grid_thw.extend(mm_input["video_grid_thw"].tolist())
                if mm_input.get("second_per_grid_ts") is not None:
                    second_per_grid_ts.extend(mm_input["second_per_grid_ts"])

            hf_config = self.model_config.hf_config

            self.requests[req_id].mrope_positions, \
                self.requests[req_id].mrope_position_delta = \
                MRotaryEmbedding.get_input_positions_tensor(
                    self.requests[req_id].prompt_token_ids,
                    hf_config=hf_config,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    second_per_grid_ts=second_per_grid_ts,
                )

        req_ids_to_add.append(req_id)

    # Update the states of the running/resumed requests.
    for req_data in scheduler_output.scheduled_cached_reqs:
        req_id = req_data.req_id
        req_state = self.requests[req_id]

        # Update the cached states.
        num_computed_tokens = req_data.num_computed_tokens
        req_state.num_computed_tokens = num_computed_tokens
        # Add the sampled token(s) from the previous step (if any).
        # This doesn't include "unverified" tokens like spec decode tokens.
        num_new_tokens = (num_computed_tokens + len(req_data.new_token_ids) -
                          req_state.num_tokens)
        if num_new_tokens == 1:
            # Avoid slicing list in most common case.
            req_state.output_token_ids.append(req_data.new_token_ids[-1])
        elif num_new_tokens > 0:
            req_state.output_token_ids.extend(
                req_data.new_token_ids[-num_new_tokens:])
        # Update the block IDs.
        if not req_data.resumed_from_preemption:
            # Append the new blocks to the existing block IDs.
            for i in range(len(self.kv_cache_config.kv_cache_groups)):
                req_state.block_ids[i].extend(req_data.new_block_ids[i])
        else:
            # The request is resumed from preemption.
            # Replace the existing block IDs with the new ones.
            req_state.block_ids = req_data.new_block_ids

        req_index = self.input_batch.req_id_to_index.get(req_id)
        if req_index is None:
            # The request is not in the persistent batch.
            # The request was either preempted and resumed later, or was not
            # scheduled in the previous step and needs to be added again.
            req_ids_to_add.append(req_id)
            continue

        # Update the persistent batch.
        self.input_batch.num_computed_tokens_cpu[req_index] = (
            num_computed_tokens)
        self.input_batch.block_table.append_row(req_data.new_block_ids,
                                                req_index)
        # Add new_token_ids to token_ids_cpu.
        start_token_index = num_computed_tokens
        end_token_index = num_computed_tokens + len(req_data.new_token_ids)
        self.input_batch.token_ids_cpu[
            req_index,
            start_token_index:end_token_index] = req_data.new_token_ids

        self.input_batch.num_tokens_no_spec[req_index] = end_token_index
        # Add spec_token_ids to token_ids_cpu.
        spec_token_ids = scheduler_output.scheduled_spec_decode_tokens.get(
            req_id, ())
        if spec_token_ids:
            start_index = end_token_index
            end_token_index += len(spec_token_ids)
            self.input_batch.token_ids_cpu[
                req_index, start_index:end_token_index] = spec_token_ids
        # NOTE(woosuk): `num_tokens` here may include spec decode tokens.
        self.input_batch.num_tokens[req_index] = end_token_index

    # Check if the batch has changed. If not, we can skip copying the
    # sampling metadata from CPU to GPU.
    batch_changed = len(removed_req_indices) > 0 or len(req_ids_to_add) > 0

    # Add the new or resumed requests to the persistent batch.
    # The smaller empty indices are filled first.
    removed_req_indices = sorted(removed_req_indices, reverse=True)
    for req_id in req_ids_to_add:
        req_state = self.requests[req_id]
        req_index = removed_req_indices.pop() if removed_req_indices else None
        self.input_batch.add_request(req_state, req_index)

    # Condense the batched states if there are empty indices.
    if removed_req_indices:
        self.input_batch.condense(removed_req_indices)

    batch_reordered = self._may_reorder_batch(scheduler_output)

    if batch_changed or batch_reordered:
        self.input_batch.refresh_sampling_metadata()


def wrapper_gpu_model_runner_execute_model(func):

    def new_func(*args, **kwargs):
        self = args[0]
        try:
            output = func(*args, **kwargs)
            return output
        except Exception:
            exc_info = traceback.format_exc()
            logger.warning("Caught exception when processing req_ids %s:\n%s",
                           self.input_batch.req_ids, exc_info)
            return ModelRunnerOutput(
                req_ids=self.input_batch.req_ids,
                req_id_to_index=self.input_batch.req_id_to_index,
                sampled_token_ids=None,
                spec_token_ids=None,
                logprobs=None,
                prompt_logprobs_dict={},
            )

    return new_func


def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
    block_size = self.vllm_config.cache_config.block_size
    use_mla = self.vllm_config.model_config.use_mla
    fa3_quant = self.vllm_config.quant_config.fa3_quant \
                    if self.vllm_config.quant_config else False
    fa3_quant_layer = self.vllm_config.quant_config.fa3_quant_layer \
                        if self.vllm_config.quant_config else set()
    kv_cache_spec: dict[str, KVCacheSpec] = {}
    attn_layers = get_layers_from_vllm_config(self.vllm_config,
                                              AttentionWrapper)
    for layer_name, attn_module in attn_layers.items():
        """
        vllm-mindspore AttentionWrapper is not an Attention isinstance
        assert isinstance(attn_module, Attention)
        """
        if attn_module.attn_type == AttentionType.DECODER:
            if attn_module.sliding_window is not None:
                kv_cache_spec[layer_name] = SlidingWindowSpec(
                    block_size=block_size,
                    num_kv_heads=attn_module.num_kv_heads,
                    head_size=attn_module.head_size,
                    dtype=self.kv_cache_dtype,
                    sliding_window=attn_module.sliding_window,
                    use_mla=use_mla)
            else:
                kv_cache_dtype = self.kv_cache_dtype
                is_fa3_quant_layer = int(layer_name) in fa3_quant_layer
                if fa3_quant and not is_fa3_quant_layer:
                    kv_cache_dtype = self.vllm_config.model_config.dtype
                if fa3_quant:
                    '''
                    fa3_quant_layer k_cache is int8, v_cache is bfloat16
                    page_size_bytes is block_size * num_kv_heads *
                    (ctkv_nope_dim * int8(1 bytes)
                    + qk_rope_dim * float16(2 bytes))
                    so need the MLAQuantFullAttentionSpec, which is a new
                    AttentionSpec.
                    and we also need the MLAQuantFullAttentionSpec for no fa3
                    quant.
                    if we have two different AttentionSpec, the
                    len(kv_cache_config.kv_cache_groups) is 2, and the
                    get_kv_cache_coordinator function return
                    HybridKVCacheCoordinator. in this coordinator,
                    the block_pool will be used by two AttentionManager.
                    and if we not change the logic of HybridKVCacheCoordinator,
                    the block pool will be allocate twice every time by two
                    AttentionManager. this will double the gpu utilization.

                    In our fa_quant scene, although we have
                    two paged size in different layer, but the block_id and
                    block table is same of different layer. so we only
                    need one AttentionManager.
                    '''

                    kv_cache_spec[layer_name] = MLAQuantFullAttentionSpec(
                        block_size=block_size,
                        num_kv_heads=attn_module.num_kv_heads,
                        head_size=attn_module.head_size,
                        dtype=kv_cache_dtype,
                        use_mla=use_mla,
                        fa3_quant=is_fa3_quant_layer)
                else:
                    kv_cache_spec[layer_name] = FullAttentionSpec(
                        block_size=block_size,
                        num_kv_heads=attn_module.num_kv_heads,
                        head_size=attn_module.head_size,
                        dtype=kv_cache_dtype,
                        use_mla=use_mla)
        elif attn_module.attn_type in (AttentionType.ENCODER,
                                       AttentionType.ENCODER_ONLY):
            # encoder-only attention does not need KV cache.
            continue
        elif attn_module.attn_type == AttentionType.ENCODER_DECODER:
            raise NotImplementedError
        else:
            raise ValueError(
                f"Unknown attention type: {attn_module.attn_type}")

    return kv_cache_spec


def _calc_mrope_positions(self, scheduler_output):
    mrope_pos_ptr = 0
    for index, req_id in enumerate(self.input_batch.req_ids):
        req = self.requests[req_id]
        assert req.mrope_positions is not None

        num_computed_tokens = \
            self.input_batch.num_computed_tokens_cpu[index]
        num_scheduled_tokens = \
            scheduler_output.num_scheduled_tokens[req_id]
        num_prompt_tokens = len(req.prompt_token_ids)

        if num_computed_tokens + num_scheduled_tokens > num_prompt_tokens:
            prompt_part_len = max(0, num_prompt_tokens - num_computed_tokens)
            completion_part_len = max(0,
                                      num_scheduled_tokens - prompt_part_len)
        else:
            prompt_part_len = num_scheduled_tokens
            completion_part_len = 0

        assert num_scheduled_tokens == prompt_part_len + completion_part_len

        if prompt_part_len > 0:
            # prompt's mrope_positions are pre-computed
            # gpu is number or tensor, but we are numpy, so we transform to int
            dst_start = int(mrope_pos_ptr)
            dst_end = int(mrope_pos_ptr + prompt_part_len)
            src_start = int(num_computed_tokens)
            src_end = int(num_computed_tokens + prompt_part_len)

            self.mrope_positions_cpu[:, dst_start:dst_end] = \
                req.mrope_positions[:,src_start:src_end]

            mrope_pos_ptr += prompt_part_len

        if completion_part_len > 0:
            # compute completion's mrope_positions on-the-fly
            dst_start = mrope_pos_ptr
            dst_end = mrope_pos_ptr + completion_part_len

            self.mrope_positions_cpu[:, dst_start:dst_end] = \
                MRotaryEmbedding.get_next_input_positions_tensor(
                    req.mrope_position_delta,
                    context_len=num_computed_tokens +
                    prompt_part_len,
                    seq_len=num_computed_tokens +
                    prompt_part_len +
                    completion_part_len,
                )

            mrope_pos_ptr += completion_part_len


def get_dp_padding(self, num_tokens: int):
    # Skip unnecessary padding processes to ensure the shape consistency
    # of model_inputs. Shape of `input_ids` and `positions` will be
    # padded based on `num_tokens_across_dp`, while the model only accepts
    # inputs with actual shape.
    return 0, None
