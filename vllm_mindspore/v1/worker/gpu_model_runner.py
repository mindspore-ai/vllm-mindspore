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

import time
import traceback
from typing import Any, Optional, Union, cast

import mindspore as ms
import numpy as np
import torch
import vllm.envs as envs
from mindspore import Generator as msGenerator
from mindspore import Tensor, mint, mutable
from typing_extensions import TypeAlias
from vllm.attention import AttentionType
from vllm.config import (CompilationLevel, CUDAGraphMode, VllmConfig,
                         get_layers_from_vllm_config)
from vllm.distributed.parallel_state import get_pp_group
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces_base import VllmModelForPooling
from vllm.sampling_params import SamplingType
from vllm.utils import round_up
from vllm.v1.attention.backends.flash_attn import AttentionMetadata
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder
from vllm.v1.attention.backends.utils import (CommonAttentionMetadata,
                                              split_attn_metadata)
from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheConfig,
                                        KVCacheSpec, MLAAttentionSpec,
                                        SlidingWindowSpec,
                                        UniformTypeKVCacheSpecs)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.spec_decode.eagle import EagleProposer
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.worker.gpu_input_batch import CachedRequestState
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.worker.ubatch_splitting import ubatch_split
from vllm.v1.worker.ubatch_utils import UBatchSlices
from vllm.v1.worker.utils import bind_kv_cache

from vllm_mindspore.model_executor.layers.rotary_embedding import (
    InferMRotaryEmbedding as MRotaryEmbedding)
from vllm_mindspore.model_executor.models.model_base import AttentionWrapper
from vllm_mindspore.model_executor.models.utils import is_use_ringmla
from vllm_mindspore.utils import (create_kv_cache, get_dtype_size,
                                  get_valid_dtype, is_310p)
from vllm_mindspore.v1.attention.backends.ms_attn import (
    MsCommonAttentionMetadata)
from vllm_mindspore.v1.kv_cache_interface import MLAQuantFullAttentionSpec

logger = init_logger(__name__)

AttnMetadataDict: TypeAlias = dict[str, AttentionMetadata]
# list when ubatching is enabled
PerLayerAttnMetadata: TypeAlias = Union[list[AttnMetadataDict],
                                        AttnMetadataDict]


class FakeStream:

    def __init__(self, *args, **kwargs):
        pass


_original_init = GPUModelRunner.__init__


# Prevent excessive event creation caused by multi-stream initialization.
# We use a FakeStream to bypass the stream creation in the original `__init__`.
# TODO: Remove this patch when the issue is resolved.
def gpu_model_runner_init(
    self,
    vllm_config: VllmConfig,
    device: torch.device,
):
    real_stream = torch.cuda.Stream
    try:
        if not vllm_config.scheduler_config.async_scheduling:
            torch.cuda.Stream = FakeStream
        _original_init(self, vllm_config, device)
    finally:
        torch.cuda.Stream = real_stream


def _to_list(self, sampled_token_ids: torch.Tensor) -> list[list[int]]:
    """
    Directly return tolist() to avoid create synchronous event
    """
    return sampled_token_ids.tolist()


def _prepare_inputs(
    self,
    scheduler_output,
) -> tuple[PerLayerAttnMetadata, Tensor, Optional[SpecDecodeMetadata],
           np.ndarray, Optional[CommonAttentionMetadata], int,
           Optional[UBatchSlices], Optional[Tensor]]:
    """
    :return: tuple[
        attn_metadata: layer-to-attention_metadata mapping,
        logits_indices, spec_decode_metadata
    ]
    """
    total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
    assert total_num_scheduled_tokens > 0
    num_reqs = self.input_batch.num_reqs
    assert num_reqs > 0

    # vllm-mindspore aclgraph only support pure decode
    self.pure_decode = num_reqs == total_num_scheduled_tokens

    # OPTIMIZATION: Start copying the block table first.
    # This way, we can overlap the copy with the following CPU operations.
    self.input_batch.block_table.commit_block_table(num_reqs)

    # Get the number of scheduled tokens for each request.
    req_ids = self.input_batch.req_ids
    tokens = [scheduler_output.num_scheduled_tokens[i] for i in req_ids]
    num_scheduled_tokens = np.array(tokens, dtype=np.int32)
    max_num_scheduled_tokens = max(tokens)

    # Get request indices.
    # E.g., [2, 5, 3] -> [0, 0, 1, 1, 1, 1, 1, 2, 2, 2]
    req_indices = np.repeat(self.arange_np[:num_reqs], num_scheduled_tokens)

    # cu_num_tokens: [2, 5, 3] -> [2, 7, 10]
    # E.g., arange: [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
    cu_num_tokens, arange = self._get_cumsum_and_arange(num_scheduled_tokens)

    # Get positions.
    positions_np = self.positions.np[:total_num_scheduled_tokens]
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

    # NOTE(woosuk): We use torch.index_select instead of np.take here
    # because torch.index_select is much faster than np.take for large
    # tensors.
    # vllm-mindspore begin
    self.input_ids.np[:total_num_scheduled_tokens] = np.take(
        self.input_batch.token_ids_cpu.ravel(), token_indices, 0)
    # vllm-mindspore end

    if self.enable_prompt_embeds:
        # vllm-mindspore begin
        self.is_token_ids.cpu[:total_num_scheduled_tokens] = ms.from_numpy(
            np.take(self.input_batch.is_token_ids.ravel(), token_indices, 0))
        # vllm-mindspore end

    # Because we did not pre-allocate a massive prompt_embeds CPU tensor on
    # the InputBatch, we need to fill in the prompt embeds into the expected
    # spots in the GpuModelRunner's pre-allocated prompt_embeds tensor.
    if self.input_batch.req_prompt_embeds:
        output_idx = 0
        for req_idx in range(num_reqs):
            num_sched = num_scheduled_tokens[req_idx]

            # Skip if this request doesn't have embeddings
            if req_idx not in self.input_batch.req_prompt_embeds:
                output_idx += num_sched
                continue

            # Skip if no tokens scheduled
            if num_sched <= 0:
                output_idx += num_sched
                continue

            req_embeds = self.input_batch.req_prompt_embeds[req_idx]
            start_pos = self.input_batch.num_computed_tokens_cpu[req_idx]

            # Skip if trying to read beyond available embeddings
            if start_pos >= req_embeds.shape[0]:
                output_idx += num_sched
                continue

            # Copy available embeddings
            end_pos = start_pos + num_sched
            actual_end = min(end_pos, req_embeds.shape[0])
            actual_num_sched = actual_end - start_pos

            if actual_num_sched > 0:
                self.inputs_embeds.cpu[output_idx:output_idx +
                                       actual_num_sched].copy_(
                                           req_embeds[start_pos:actual_end])

            output_idx += num_sched

    self.input_batch.block_table.compute_slot_mapping(req_indices,
                                                      positions_np)
    self.input_batch.block_table.commit_slot_mapping(
        total_num_scheduled_tokens)

    num_tokens_unpadded = scheduler_output.total_num_scheduled_tokens
    num_tokens_padded = num_tokens_unpadded
    num_reqs_padded = num_reqs
    if self.pure_decode:
        num_tokens_padded = num_tokens_unpadded + self.get_local_padding(
            num_tokens_unpadded)
        num_reqs_padded = num_tokens_padded

    # Prepare the attention metadata.
    self.query_start_loc.np[0] = 0
    self.query_start_loc.np[1:num_reqs + 1] = cu_num_tokens
    # Note: pad query_start_loc to be non-decreasing, as kernels
    # like FlashAttention requires that
    self.query_start_loc.np[num_reqs + 1:].fill(cu_num_tokens[-1])
    q_seq_lens_np = np.diff(self.query_start_loc.np[:num_reqs_padded + 1])

    uniform_decode = \
        (max_num_scheduled_tokens == self.uniform_decode_query_len) and \
        (total_num_scheduled_tokens == num_reqs * max_num_scheduled_tokens)
    ubatch_slices, num_tokens_after_padding = \
        ubatch_split(num_scheduled_tokens,
                     num_tokens_unpadded,
                     num_tokens_padded,
                     uniform_decode=uniform_decode,
                     vllm_config=self.vllm_config)

    self.seq_lens.np[:num_reqs] = (
        self.input_batch.num_computed_tokens_cpu[:num_reqs] +
        num_scheduled_tokens)
    # Fill unused with 0 for full cuda graph mode.
    self.seq_lens.np[num_reqs_padded:].fill(0)
    self.seq_lens.copy_to_gpu()
    seq_lens = self.seq_lens.gpu[:num_reqs_padded]
    max_seq_len = self.seq_lens.np[:num_reqs_padded].max().item()

    num_tokens = [
        self.requests[r].num_tokens for r in self.input_batch.req_ids
    ]
    num_tokens_np = np.array(num_tokens, dtype=np.int32)

    # Record the index of requests that should not be sampled,
    # so that we could clear the sampled tokens before returning
    discard_requests_mask = self.seq_lens.np[:num_reqs] < num_tokens_np
    discard_request_indices = np.nonzero(discard_requests_mask)[0]
    self.num_discarded_requests = len(discard_request_indices)
    self.discard_request_indices.np[:self.num_discarded_requests] = (
        discard_request_indices)

    self.discard_request_indices.copy_to_gpu(self.num_discarded_requests)

    # Copy the tensors to the GPU.
    self._prepare_input_ids(total_num_scheduled_tokens, cu_num_tokens)

    if self.uses_mrope:
        # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
        self.mrope_positions.gpu[:, :total_num_scheduled_tokens].copy_(
            self.mrope_positions.cpu[:, :total_num_scheduled_tokens],
            non_blocking=True)
    else:
        # Common case (1D positions)
        self.positions.copy_to_gpu(total_num_scheduled_tokens)

    use_spec_decode = len(scheduler_output.scheduled_spec_decode_tokens) > 0
    if not use_spec_decode:
        # NOTE(woosuk): Due to chunked prefills, the batch may contain
        # partial requests. While we should not sample any token
        # from these partial requests, we do so for simplicity.
        # We will ignore the sampled tokens from the partial requests.
        # TODO: Support prompt logprobs.
        # vllm-mindspore begin
        query_start_loc = ms.from_numpy(self.query_start_loc.np[:num_reqs + 1])
        logits_indices = query_start_loc[1:] - 1
        # vllm-mindspore end
        num_draft_tokens = None
        spec_decode_metadata = None
    else:
        # Get the number of draft tokens for each request.
        # Iterate over the dictionary rather than all requests since not all
        # requests have draft tokens.
        num_draft_tokens = np.zeros(num_reqs, dtype=np.int32)
        # For chunked prefills, use -1 as mask rather than 0, as guided
        # decoding may rollback speculative tokens.
        num_decode_draft_tokens = np.full(num_reqs, -1, dtype=np.int32)
        for req_id, draft_token_ids in (
                scheduler_output.scheduled_spec_decode_tokens.items()):
            req_idx = self.input_batch.req_id_to_index[req_id]
            num_draft_tokens[req_idx] = len(draft_token_ids)
            num_decode_draft_tokens[req_idx] = (len(draft_token_ids) if (
                self.input_batch.num_computed_tokens_cpu[req_idx]
                >= self.input_batch.num_prompt_tokens[req_idx]) else -1)

        spec_decode_metadata = self._calc_spec_decode_metadata(
            num_draft_tokens, cu_num_tokens)
        logits_indices = spec_decode_metadata.logits_indices
        # For DECODE only cuda graph of some attention backends (e.g., GDN).
        self.num_decode_draft_tokens.np[:num_reqs] = num_decode_draft_tokens
        self.num_decode_draft_tokens.np[num_reqs:].fill(-1)
        self.num_decode_draft_tokens.copy_to_gpu()

    logits_indices_padded = None
    if self.cache_config.kv_sharing_fast_prefill:
        logits_indices_padded = self._prepare_kv_sharing_fast_prefill(
            logits_indices)

    attn_metadata: PerLayerAttnMetadata = {}
    if ubatch_slices is not None:
        attn_metadata = [dict() for _ in range(len(ubatch_slices))]

    # Used in the below loop.
    spec_decode_common_attn_metadata = None
    if use_spec_decode:
        self.num_accepted_tokens.np[:num_reqs] = (
            self.input_batch.num_accepted_tokens_cpu[:num_reqs])
        self.num_accepted_tokens.np[num_reqs:].fill(1)
        self.num_accepted_tokens.copy_to_gpu()

    # Prepare the attention metadata for each KV cache group and make layers
    # in the same group share the same metadata.
    for kv_cache_group_id, kv_cache_group_spec in enumerate(
            self.kv_cache_config.kv_cache_groups):
        encoder_seq_lens = self._get_encoder_seq_lens(
            scheduler_output, kv_cache_group_spec.kv_cache_spec, num_reqs)

        blk_table = self.input_batch.block_table[kv_cache_group_id]
        blk_table_tensor = blk_table.get_device_tensor(num_reqs_padded)
        slot_mapping_np = blk_table.slot_mapping.np[:num_tokens_padded]

        # Fill unused with -1. Needed for reshape_and_cache in full cuda
        # graph mode.
        blk_table.slot_mapping.gpu[total_num_scheduled_tokens:].fill_(-1)
        num_common_prefix_blocks = (
            scheduler_output.num_common_prefix_blocks[kv_cache_group_id])

        common_attn_metadata = MsCommonAttentionMetadata(
            query_start_loc=None,
            query_start_loc_cpu=None,
            q_seq_lens_np=q_seq_lens_np,
            seq_lens=seq_lens,
            seq_lens_cpu=None,
            seq_lens_np=self.seq_lens.np[:num_reqs_padded],
            num_computed_tokens_cpu=None,
            num_computed_tokens_np=self.input_batch.
            num_computed_tokens_cpu[:num_reqs],
            num_reqs=num_reqs,
            num_actual_tokens=total_num_scheduled_tokens,
            max_query_len=max_num_scheduled_tokens,
            max_seq_len=max_seq_len,
            block_table_tensor=blk_table_tensor,
            slot_mapping=None,
            slot_mapping_np=slot_mapping_np,
            logits_indices_padded=logits_indices_padded,
            num_logits_indices=logits_indices.size(0),
            causal=True,
            encoder_seq_lens=encoder_seq_lens,
        )

        if (self.speculative_config
                and spec_decode_common_attn_metadata is None):
            if isinstance(self.drafter, EagleProposer):
                if (self.drafter.attn_layer_names[0]
                        in kv_cache_group_spec.layer_names):
                    spec_decode_common_attn_metadata = common_attn_metadata
            else:
                spec_decode_common_attn_metadata = common_attn_metadata

        for attn_group in self.attn_groups[kv_cache_group_id]:
            # Prepare for cascade attention if enabled & beneficial.
            common_prefix_len = 0
            builder = attn_group.get_metadata_builder()
            if self.cascade_attn_enabled:
                common_prefix_len = self._compute_cascade_attn_prefix_len(
                    num_scheduled_tokens,
                    num_common_prefix_blocks,
                    attn_group.kv_cache_spec,
                    builder,
                )

            extra_attn_metadata_args = {}
            if use_spec_decode and isinstance(builder,
                                              GDNAttentionMetadataBuilder):
                extra_attn_metadata_args = dict(
                    num_accepted_tokens=self.num_accepted_tokens.
                    gpu[:num_reqs],
                    num_draft_tokens=self.num_draft_tokens.gpu[:num_reqs],
                )

            if ubatch_slices is not None:
                common_attn_metadata_list = split_attn_metadata(
                    ubatch_slices, common_attn_metadata)
                for ubid, common_attn_metadata in enumerate(
                        common_attn_metadata_list):
                    attn_metadata_i = (attn_group.get_metadata_builder(
                        ubatch_id=ubid).build(
                            common_prefix_len=common_prefix_len,
                            common_attn_metadata=common_attn_metadata))
                    for layer_name in kv_cache_group_spec.layer_names:
                        assert type(attn_metadata) is list
                        attn_metadata[ubid][layer_name] = attn_metadata_i
            else:
                assert isinstance(attn_metadata, dict)
                attn_metadata_i = builder.build(
                    common_prefix_len=common_prefix_len,
                    common_attn_metadata=common_attn_metadata,
                    **extra_attn_metadata_args)
                for layer_name in attn_group.layer_names:
                    attn_metadata[layer_name] = attn_metadata_i

    # Hot-Swap lora model
    if self.lora_config:
        self.set_active_loras(self.input_batch, num_scheduled_tokens)

    return (attn_metadata, logits_indices, spec_decode_metadata,
            num_scheduled_tokens, spec_decode_common_attn_metadata,
            max_num_scheduled_tokens, ubatch_slices, num_tokens_after_padding)


def _get_num_input_tokens(self, num_scheduled_tokens: int) -> int:
    if hasattr(self, "pure_decode") and not self.pure_decode:
        return num_scheduled_tokens

    if (self.compilation_config.cudagraph_mode != CUDAGraphMode.NONE
            and not envs.VLLM_DISABLE_PAD_FOR_CUDAGRAPH
            and hasattr(self, "cudagraph_batch_sizes")
            and self.cudagraph_batch_sizes
            and num_scheduled_tokens <= self.cudagraph_batch_sizes[-1]):
        # Use CUDA graphs.
        # Add padding to the batch size.
        return self.vllm_config.pad_for_cudagraph(num_scheduled_tokens)

    # Eager mode.
    # Pad tokens to multiple of tensor_parallel_size when
    # enabled collective fusion for SP
    tp_size = self.vllm_config.parallel_config.tensor_parallel_size
    if (self.compilation_config.pass_config.enable_sequence_parallelism
            and tp_size > 1):
        return round_up(num_scheduled_tokens, tp_size)
    return num_scheduled_tokens


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

        target_dtype = get_valid_dtype(kv_cache_spec.dtype)

        num_blocks = kv_cache_tensor.size // kv_cache_spec.page_size_bytes
        kv_cache_shape = (num_blocks, kv_cache_spec.block_size,
                          kv_cache_spec.num_kv_heads, kv_cache_spec.head_size)

        reshaped_layer_tensors = []
        coef = 1 if isinstance(kv_cache_spec,
                               (MLAAttentionSpec,
                                MLAQuantFullAttentionSpec)) else 2
        for _ in range(coef):
            reshaped_layer_tensors.append(
                create_kv_cache(kv_cache_shape, target_dtype))

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

    assert len(kv_cache_config.kv_cache_groups) == 1
    assert isinstance(kv_cache_config.kv_cache_groups[0].kv_cache_spec,
                      UniformTypeKVCacheSpecs)
    per_layer_specs = kv_cache_config.kv_cache_groups[
        0].kv_cache_spec.kv_cache_specs
    # fa3 quant layer target_dtype is int8
    # no fa3 quant layer target_dtype is bfloat16
    fa3_quant = getattr(self.vllm_config.quant_config, "fa3_quant", False)
    fa3_quant_layer: set[int] = getattr(self.vllm_config.quant_config,
                                        "fa3_quant_layer", set())
    kv_lora_rank = getattr(self.vllm_config.model_config.hf_text_config,
                           'kv_lora_rank', 0)
    qk_rope_head_dim = getattr(self.vllm_config.model_config.hf_text_config,
                               'qk_rope_head_dim', 0)
    for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
        if not kv_cache_tensor.shared_by:
            continue
        rep_layer_name = kv_cache_tensor.shared_by[0]
        kv_cache_spec = per_layer_specs[rep_layer_name]
        is_fa3_quant_layer = fa3_quant and int(rep_layer_name) \
            in fa3_quant_layer
        num_blocks = kv_cache_tensor.size // kv_cache_spec.page_size_bytes
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

    all_layers = set(per_layer_specs.keys())
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
    dtype = kv_cache_spec.dtype

    # Determine the number of tensors in kv_cache
    # using the type of kv_cache_spec.
    coef = 1 if isinstance(kv_cache_spec, (MLAAttentionSpec,
                                           MLAQuantFullAttentionSpec)) else 2

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
    for group in self._kv_cache_spec_attn_group_iterator():
        kv_cache_spec = group.kv_cache_spec
        attn_backend = group.backend

        # Determine the number of tensors in kv_cache
        # using the type of kv_cache_spec.
        coef = 1 if isinstance(kv_cache_spec,
                               (MLAAttentionSpec,
                                MLAQuantFullAttentionSpec)) else 2

        for layer_name in group.layer_names:
            if layer_name in self.runner_only_attn_layers:
                continue
            raw_tensor = kv_cache_raw_tensors[layer_name]
            target_dtype = get_valid_dtype(kv_cache_spec.dtype)
            dtype_size = get_dtype_size(target_dtype)
            num_blocks = \
                (raw_tensor[0].numel() if not use_ringmla else \
                (raw_tensor[0].numel() + raw_tensor[1].numel())) * \
                coef * dtype_size // kv_cache_spec.page_size_bytes
            if isinstance(kv_cache_spec,
                          (FullAttentionSpec, MLAQuantFullAttentionSpec)):
                kv_cache_shape = attn_backend.get_kv_cache_shape(
                    num_blocks,
                    kv_cache_spec.block_size,
                    kv_cache_spec.num_kv_heads,
                    kv_cache_spec.head_size,
                    cache_dtype_str=self.cache_config.cache_dtype)
                try:
                    kv_cache_stride_order = \
                        attn_backend.get_kv_cache_stride_order()
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

    # Set up cross-layer KV cache sharing
    for layer_name, target_layer_name in self.shared_kv_cache_layers.items():
        logger.debug("%s reuses KV cache of %s", layer_name, target_layer_name)
        kv_caches[layer_name] = kv_caches[target_layer_name]

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
    # Remove the finished requests from the persistent batch.
    # NOTE(woosuk): There could be an edge case where finished_req_ids and
    # scheduled_req_ids overlap. This happens when a request is aborted and
    # then resubmitted with the same ID. In this case, we treat them as two
    # distinct requests - clearing the cached states for the first request
    # and handling the second as a new request.
    for req_id in scheduler_output.finished_req_ids:
        self.input_batch.remove_request(req_id)

    # Free the cached encoder outputs.
    for mm_hash in scheduler_output.free_encoder_mm_hashes:
        self.encoder_cache.pop(mm_hash, None)

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
        self.input_batch.remove_request(req_id)

    reqs_to_add: list[CachedRequestState] = []
    # Add new requests to the cached states.
    for new_req_data in scheduler_output.scheduled_new_reqs:
        req_id = new_req_data.req_id
        sampling_params = new_req_data.sampling_params
        pooling_params = new_req_data.pooling_params

        if sampling_params and \
            sampling_params.sampling_type == SamplingType.RANDOM_SEED:
            # vllm-mindspore begin
            generator = msGenerator()
            # vllm-mindspore end
            generator.manual_seed(sampling_params.seed)
        else:
            generator = None

        if self.is_pooling_model:
            assert pooling_params is not None
            task = pooling_params.task
            assert task is not None, "You did not set `task` in the API"

            model = cast(VllmModelForPooling, self.get_model())
            to_update = model.pooler.get_pooling_updates(task)
            to_update.apply(pooling_params)

        req_state = CachedRequestState(
            req_id=req_id,
            prompt_token_ids=new_req_data.prompt_token_ids,
            prompt_embeds=new_req_data.prompt_embeds,
            mm_features=new_req_data.mm_features,
            sampling_params=sampling_params,
            pooling_params=pooling_params,
            generator=generator,
            block_ids=new_req_data.block_ids,
            num_computed_tokens=new_req_data.num_computed_tokens,
            output_token_ids=[],
            lora_request=new_req_data.lora_request,
        )
        self.requests[req_id] = req_state

        # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
        if self.uses_mrope:
            self._init_mrope_positions(req_state)

        reqs_to_add.append(req_state)

    # Update the states of the running/resumed requests.
    is_last_rank = get_pp_group().is_last_rank
    req_data = scheduler_output.scheduled_cached_reqs
    for i, req_id in enumerate(req_data.req_ids):
        req_state = self.requests[req_id]
        num_computed_tokens = req_data.num_computed_tokens[i]
        new_block_ids = req_data.new_block_ids[i]
        resumed_from_preemption = req_data.resumed_from_preemption[i]

        # Update the cached states.
        req_state.num_computed_tokens = num_computed_tokens

        if not is_last_rank:
            # When using PP, the scheduler sends the sampled tokens back,
            # because there's no direct communication between the first-
            # stage worker and the last-stage worker.
            new_token_ids = req_data.new_token_ids[i]
            # Add the sampled token(s) from the previous step (if any).
            # This doesn't include "unverified" tokens like spec tokens.
            num_new_tokens = (num_computed_tokens + len(new_token_ids) -
                              req_state.num_tokens)
            if num_new_tokens == 1:
                # Avoid slicing list in most common case.
                req_state.output_token_ids.append(new_token_ids[-1])
            elif num_new_tokens > 0:
                req_state.output_token_ids.extend(
                    new_token_ids[-num_new_tokens:])

        # Update the block IDs.
        if not resumed_from_preemption:
            if new_block_ids is not None:
                # Append the new blocks to the existing block IDs.
                for block_ids, new_ids in zip(req_state.block_ids,
                                              new_block_ids):
                    block_ids.extend(new_ids)
        else:
            assert new_block_ids is not None
            # The request is resumed from preemption.
            # Replace the existing block IDs with the new ones.
            req_state.block_ids = new_block_ids

        req_index = self.input_batch.req_id_to_index.get(req_id)
        if req_index is None:
            # The request is not in the persistent batch.
            # The request was either preempted and resumed later, or was not
            # scheduled in the previous step and needs to be added again.
            reqs_to_add.append(req_state)
            continue

        # Update the persistent batch.
        self.input_batch.num_computed_tokens_cpu[req_index] = (
            num_computed_tokens)
        if new_block_ids is not None:
            self.input_batch.block_table.append_row(new_block_ids, req_index)

        # For the last rank, we don't need to update the token_ids_cpu
        # because the sampled tokens are already cached.
        if not is_last_rank:
            # Add new_token_ids to token_ids_cpu.
            start_token_index = num_computed_tokens
            end_token_index = num_computed_tokens + len(new_token_ids)
            self.input_batch.token_ids_cpu[
                req_index, start_token_index:end_token_index] = new_token_ids
            self.input_batch.num_tokens_no_spec[req_index] = end_token_index
            self.input_batch.num_tokens[req_index] = end_token_index

        # Add spec_token_ids to token_ids_cpu.
        spec_token_ids = (scheduler_output.scheduled_spec_decode_tokens.get(
            req_id, ()))
        if spec_token_ids:
            num_spec_tokens = len(spec_token_ids)
            start_index = self.input_batch.num_tokens_no_spec[req_index]
            end_token_index = start_index + num_spec_tokens
            self.input_batch.token_ids_cpu[
                req_index, start_index:end_token_index] = spec_token_ids
            # NOTE(woosuk): `num_tokens` here may include spec tokens.
            self.input_batch.num_tokens[req_index] += num_spec_tokens

    # Add the new or resumed requests to the persistent batch.
    # The smaller empty indices are filled first.
    for request in reqs_to_add:
        self.input_batch.add_request(request)

    # Condense the batched states if there are gaps left by removed requests
    self.input_batch.condense()
    # Allow attention backend to reorder the batch, potentially
    self._may_reorder_batch(scheduler_output)
    # Refresh batch metadata with any pending updates.
    self.input_batch.refresh_metadata()


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
                logprobs=None,
                prompt_logprobs_dict={},
                pooler_output=[])

    return new_func


def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
    block_size = self.vllm_config.cache_config.block_size
    use_mla = self.vllm_config.model_config.use_mla
    fa3_quant = getattr(self.vllm_config.quant_config, "fa3_quant", False)
    fa3_quant_layer: set[int] = getattr(self.vllm_config.quant_config,
                                        "fa3_quant_layer", set())
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
                )
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
                        fa3_quant=is_fa3_quant_layer)
                elif use_mla:
                    kv_cache_spec[layer_name] = MLAAttentionSpec(
                        block_size=block_size,
                        num_kv_heads=attn_module.num_kv_heads,
                        head_size=attn_module.head_size,
                        dtype=self.kv_cache_dtype)
                else:
                    kv_cache_spec[layer_name] = FullAttentionSpec(
                        block_size=block_size,
                        num_kv_heads=attn_module.num_kv_heads,
                        head_size=attn_module.head_size,
                        dtype=kv_cache_dtype)
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

            self.mrope_positions.cpu[:, dst_start:dst_end] = \
                req.mrope_positions[:,src_start:src_end]

            mrope_pos_ptr += prompt_part_len

        if completion_part_len > 0:
            # compute completion's mrope_positions on-the-fly
            dst_start = mrope_pos_ptr
            dst_end = mrope_pos_ptr + completion_part_len

            MRotaryEmbedding.get_next_input_positions_tensor(
                out=self.mrope_positions.cpu,
                out_offset=dst_start,
                mrope_position_delta=req.mrope_position_delta,
                context_len=num_computed_tokens + prompt_part_len,
                num_new_tokens=completion_part_len,
            )

            mrope_pos_ptr += completion_part_len


def get_dp_padding(self, num_tokens: int):
    # Skip unnecessary padding processes to ensure the shape consistency
    # of model_inputs. Shape of `input_ids` and `positions` will be
    # padded based on `num_tokens_across_dp`, while the model only accepts
    # inputs with actual shape.
    return 0, None


def _aclgraph_capture_dummy_run(
    self: GPUModelRunner,
    num_tokens: int,
    skip_attn: bool = True,
):
    # Padding for DP
    num_pad, num_tokens_across_dp = self.get_dp_padding(num_tokens=num_tokens)
    num_tokens += num_pad

    # Set num_scheduled_tokens based on num_tokens and max_num_seqs
    # for dummy run with LoRA so that the num_reqs collectively
    # has num_tokens in total.

    assert num_tokens <= self.scheduler_config.max_num_batched_tokens
    max_num_reqs = self.scheduler_config.max_num_seqs
    num_reqs = min(num_tokens, max_num_reqs)
    min_tokens_per_seq = num_tokens // num_reqs
    num_scheduled_tokens_list = [min_tokens_per_seq] * num_reqs
    num_scheduled_tokens_list[-1] += num_tokens % num_reqs
    assert sum(num_scheduled_tokens_list) == num_tokens
    assert len(num_scheduled_tokens_list) == num_reqs
    num_scheduled_tokens = np.array(num_scheduled_tokens_list, dtype=np.int32)
    if skip_attn:
        attn_metadata: Optional[dict[str, Any]] = None
    else:
        # Make sure max_model_len is used at the graph capture time.
        self.seq_lens_np[:num_reqs] = self.max_model_len
        self.seq_lens_np[num_reqs:] = 0
        self.seq_lens[:num_reqs].copy_(self.seq_lens_cpu[:num_reqs],
                                       non_blocking=True)

        attn_metadata = {}

        # Prepare the attention memdata for each KV cache group and make layers
        # in the same group share the same metadata.
        for kv_cache_group_id, kv_cache_group_spec in enumerate(
                self.kv_cache_config.kv_cache_groups):
            # Prepare for cascade attention if enable & beneficial.
            common_prefix_len = 0
            attn_metadata_i = (
                self.attn_metadata_builders[kv_cache_group_id].build(
                    num_reqs=num_reqs,
                    num_actual_tokens=num_tokens,
                    max_query_len=num_tokens,
                    common_prefix_len=common_prefix_len,
                ))
            # disable prefill by set max_context_len != 0
            attn_metadata_i.max_context_lens = 1
            for layer_name in kv_cache_group_spec.layer_names:
                attn_metadata[layer_name] = attn_metadata_i

    with self.maybe_dummy_run_with_lora(
            self.lora_config, num_scheduled_tokens=num_scheduled_tokens):
        model = self.model
        input_ids = self.input_ids.cpu[:num_tokens]
        inputs_embeds = None
        positions = self.positions.cpu[:num_tokens]

        if get_pp_group().is_first_rank:
            intermediate_tensors = None
        else:
            if self.intermediate_tensors is None:
                self.intermediate_tensors = (
                    self.model.make_empty_intermediate_tensors(
                        batch_size=self.max_num_tokens,
                        dtype=self.model_config.dtype,
                        device=self.device))
            intermediate_tensors = self.sync_and_slice_intermediate_tensors(
                num_tokens, None, False)

        with self.maybe_randomize_inputs(input_ids), set_forward_context(
                attn_metadata,
                self.vllm_config,
                num_tokens=num_tokens,
                num_tokens_across_dp=num_tokens_across_dp):
            outputs = model(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
            )
        hidden_states = outputs

    logit_indices = np.cumsum(num_scheduled_tokens) - 1
    return hidden_states[logit_indices]


def capture_model(self: GPUModelRunner) -> None:
    if self.compilation_config.level != CompilationLevel.PIECEWISE or \
        self.compilation_config.cudagraph_mode != CUDAGraphMode.PIECEWISE:
        logger.warning("AclGraph is disabled, "
                       "please enable it by passing -O 3.")
        return

    start_time = time.perf_counter()
    start_free_gpu_memory = torch.cuda.mem_get_info()[0]

    # Trigger aclgraph capture for specific shapes
    # Capture the large shapes first so that the smaller shapes
    # can reuse the memory poll allocated for the large shapes.

    self.cudagraph_dispatcher.initialize_cudagraph_keys(
        self.compilation_config.cudagraph_mode, self.uniform_decode_query_len)

    # because aclgraph limit, check the capture size
    max_capture_graph_size = 19
    if len(self.cudagraph_batch_sizes) > max_capture_graph_size:
        logger.warning(
            "Capture size is too much for aclgraph capture,"
            "capture %d instead", max_capture_graph_size)
        self.cudagraph_batch_sizes = self.cudagraph_batch_sizes[:
                                                                max_capture_graph_size]

    # enable mindspore graph capture
    ms.set_kernel_launch_capture(True,
                                 op_capture_skip=["moveto", "custom_mla"])
    for num_tokens in reversed(self.cudagraph_batch_sizes):
        for _ in range(
                self.vllm_config.compilation_config.cudagraph_num_of_warmups):
            self._dummy_run(num_tokens=num_tokens,
                            cudagraph_runtime_mode=CUDAGraphMode.NONE,
                            force_attention=False,
                            uniform_decode=True,
                            allow_microbatching=False,
                            skip_eplb=True,
                            remove_lora=False)
        self._dummy_run(num_tokens=num_tokens,
                        cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
                        force_attention=False,
                        uniform_decode=True,
                        allow_microbatching=False,
                        skip_eplb=True,
                        remove_lora=False)

    end_time = time.perf_counter()
    end_free_gpu_memory = torch.cuda.mem_get_info()[0]
    elapsed_time = end_time - start_time
    cuda_graph_size = start_free_gpu_memory - end_free_gpu_memory
    logger.info("Graph capturing finished in %.0f secs, took %.2f GiB",
                elapsed_time, cuda_graph_size / (1 << 30))

    # disable mindspore graph capture (captured graphs replay still work)
