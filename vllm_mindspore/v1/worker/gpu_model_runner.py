
from typing import Dict, Tuple, List
import gc
import numpy as np
import torch

from mindspore import mutable
import mindspore as ms
from vllm_mindspore.v1.attention.backends.flash_attn import (FlashAttentionMetadata,
                                                             FlashAttentionBackend,
                                                             MLABackend)
from vllm_mindspore.utils import get_valid_dtype

from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.utils import bind_kv_cache
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalKwargs
from vllm.distributed.parallel_state import get_pp_group
from vllm.utils import cdiv
from vllm.logger import init_logger
from vllm.v1.worker.gpu_input_batch import CachedRequestState
from vllm.model_executor.layers.rotary_embedding import MRotaryEmbedding
from vllm.sampling_params import SamplingType


logger = init_logger(__name__)
def _prepare_inputs(
    self,
    scheduler_output: "SchedulerOutput",
) -> Tuple[FlashAttentionMetadata, torch.Tensor]:
    total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
    assert total_num_scheduled_tokens > 0
    num_reqs = self.input_batch.num_reqs
    assert num_reqs > 0

    modified_batch = self.attn_metadata_builder.reorder_batch(
        self.input_batch, scheduler_output)
    if modified_batch:
        self.input_batch.refresh_sampling_metadata()

    # OPTIMIZATION: Start copying the block table first.
    # This way, we can overlap the copy with the following CPU operations.
    self.input_batch.block_table.commit(num_reqs)

    # Get the number of scheduled tokens for each request.
    # TODO: The Python loop can be slow. Optimize.
    num_scheduled_tokens = np.empty(num_reqs, dtype=np.int32)
    max_num_scheduled_tokens = 0
    for i, req_id in enumerate(self.input_batch.req_ids):
        num_tokens = scheduler_output.num_scheduled_tokens[req_id]
        num_scheduled_tokens[i] = num_tokens
        max_num_scheduled_tokens = max(max_num_scheduled_tokens,
                                        num_tokens)

    # Get request indices.
    # E.g., [2, 5, 3] -> [0, 0, 1, 1, 1, 1, 1, 2, 2, 2]
    req_indices = np.repeat(self.arange_np[:num_reqs],
                            num_scheduled_tokens)

    # Get batched arange.
    # E.g., [2, 5, 3] -> [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
    # Equivalent to but faster than:
    # np.concatenate([np.arange(n) for n in num_scheduled_tokens])
    # Step 1. [2, 5, 3] -> [2, 7, 10]
    cu_num_tokens = np.cumsum(num_scheduled_tokens)
    # Step 2. [2, 7, 10] -> [0, 0, 2, 2, 2, 2, 2, 7, 7, 7]
    cumsums_offsets = np.repeat(cu_num_tokens - num_scheduled_tokens,
                                num_scheduled_tokens)
    # Step 3. [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
    arange = self.arange_np[:total_num_scheduled_tokens] - cumsums_offsets

    # Get positions.
    positions_np = self.positions_np[:total_num_scheduled_tokens]
    np.add(self.input_batch.num_computed_tokens_cpu[req_indices],
            arange,
            out=positions_np)

    if self.uses_mrope:
        self._calc_mrope_positions(scheduler_output)

    if self.uses_mrope:
        # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
        self.mrope_positions[:, :total_num_scheduled_tokens].copy_(
            self.mrope_positions_cpu[:, :total_num_scheduled_tokens],
            non_blocking=True)
    else:
        self.positions[:total_num_scheduled_tokens] = torch.from_numpy(positions_np)


    # Get token indices.
    # E.g., [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
    # -> [0, 1, M, M + 1, M + 2, M + 3, M + 4, 2 * M, 2 * M + 1, 2 * M + 2]
    # where M is the max_model_len.
    token_indices = (positions_np +
                     req_indices * self.input_batch.token_ids_cpu.shape[1])

    self.input_ids[:total_num_scheduled_tokens] = torch.from_numpy(
        np.take(self.input_batch.token_ids_cpu.ravel(),
                token_indices,
                0)
    )

    # Calculate the slot mapping.
    # E.g., [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
    # -> [0, 0, K, K, K + 1, K + 1, K + 2, 2 * K, 2 * K, 2 * K + 1]
    # where K is the max_num_blocks_per_req and the block size is 2.
    # NOTE(woosuk): We can't simply use `token_indices // block_size` here
    # because M (max_model_len) is not necessarily divisible by block_size.
    block_table_indices = (req_indices * self.max_num_blocks_per_req +
                           positions_np // self.block_size)


    block_numbers = self.input_batch.block_table.block_table_np.ravel()[block_table_indices]
    block_offsets = positions_np % self.block_size
    np.add(block_numbers * self.block_size,
            block_offsets,
            out=self.slot_mapping_np[:total_num_scheduled_tokens])

    # # Prepare the attention metadata.
    self.query_start_loc_np[0] = 0
    self.query_start_loc_np[1:num_reqs + 1] = cu_num_tokens

    self.seq_lens_np[:num_reqs] = (
        self.input_batch.num_computed_tokens_cpu[:num_reqs] +
        num_scheduled_tokens)

    common_prefix_len = 0
    if self.cascade_attn_enabled:
        common_prefix_len = self._compute_cascade_attn_prefix_len(
            num_scheduled_tokens,
            scheduler_output.num_common_prefix_blocks,
        )

    attn_metadata = self.attn_metadata_builder.build(
        num_reqs=num_reqs,
        num_actual_tokens=total_num_scheduled_tokens,
        max_query_len=max_num_scheduled_tokens,
        common_prefix_len=common_prefix_len,
    )

    use_spec_decode = len(
        scheduler_output.scheduled_spec_decode_tokens) > 0
    if not use_spec_decode:
        # NOTE(woosuk): Due to chunked prefills, the batch may contain
        # partial requests. While we should not sample any token
        # from these partial requests, we do so for simplicity.
        # We will ignore the sampled tokens from the partial requests.
        # TODO: Support prompt logprobs.
        logits_indices = attn_metadata.query_start_loc[1:] - 1
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
    from mindspore import mint
    blocks = mint.empty(shape, dtype=dtype, device=device)
    return blocks

def initialize_kv_cache(self, kv_cache_config) -> None:
    """
    Initialize KV cache based on `kv_cache_config`.
    Args:
        kv_cache_config: Configuration for the KV cache, including the KV 
        cache size of each layer
    """
    if len(kv_cache_config.kv_cache_groups) > 1:
        raise NotImplementedError(
            "Hybrid models with more than one KV cache type are not "
            "supported yet.")

    kv_caches: Dict[str, torch.Tensor] = {}

    for kv_cache_group in kv_cache_config.kv_cache_groups:
        kv_cache_spec = kv_cache_group.kv_cache_spec
        for layer_name in kv_cache_group.layer_names:
            tensor_config = kv_cache_config.tensors[layer_name]
            assert tensor_config.size % kv_cache_spec.page_size_bytes == 0
            num_blocks = tensor_config.size // kv_cache_spec.page_size_bytes
            # `num_blocks` is the number of blocks the model runner can use.
            # `kv_cache_config.num_blocks` is the number of blocks that
            # KVCacheManager may allocate.
            # Since different GPUs may have different number of layers and
            # different memory capacities, `num_blocks` can be different on
            # different GPUs, and `kv_cache_config.num_blocks` is set to
            # the min of all `num_blocks`. Verify it here.
            assert num_blocks >= kv_cache_config.num_blocks
            if isinstance(kv_cache_spec, FullAttentionSpec):
                kv_cache_shape = self.attn_backend.get_kv_cache_shape(
                    num_blocks, kv_cache_spec.block_size, kv_cache_spec.num_kv_heads,
                    kv_cache_spec.head_size)
                dtype = kv_cache_spec.dtype
                dtype = get_valid_dtype(dtype)
                current_cache = []
                device_type = "CPU" if self.device.type == "cpu" else "Ascend"
                for i in range(kv_cache_shape[0]):
                    cache_blocks = create_block(
                        kv_cache_shape[1:], dtype, device=device_type
                    )
                    current_cache.append(mutable(cache_blocks))
                kv_caches[layer_name] = mutable(tuple(current_cache))
            else:
                raise NotImplementedError

    bind_kv_cache(
        kv_caches,
        self.vllm_config.compilation_config.static_forward_context,
        self.kv_caches)


def _update_states(self, scheduler_output: "SchedulerOutput") -> None:
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
    removed_req_indices: List[int] = []
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

    req_ids_to_add: List[str] = []
    # Add new requests to the cached states.
    for new_req_data in scheduler_output.scheduled_new_reqs:
        req_id = new_req_data.req_id
        sampling_params = new_req_data.sampling_params
        if sampling_params.sampling_type == SamplingType.RANDOM_SEED:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(sampling_params.seed)
        else:
            generator = None

        self.requests[req_id] = CachedRequestState(
            req_id=req_id,
            prompt_token_ids=new_req_data.prompt_token_ids,
            prompt=new_req_data.prompt,
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
                    image_grid_thw.extend(
                        mm_input["image_grid_thw"].tolist())
                if mm_input.get("video_grid_thw") is not None:
                    video_grid_thw.extend(
                        mm_input["video_grid_thw"].tolist())
                if mm_input.get("second_per_grid_ts") is not None:
                    second_per_grid_ts.extend(
                        mm_input["second_per_grid_ts"])

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
        num_new_tokens = (num_computed_tokens +
                            len(req_data.new_token_ids) -
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
            req_state.block_ids.extend(req_data.new_block_ids)
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
        start_index = (len(req_state.block_ids) -
                        len(req_data.new_block_ids))
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


    # self.input_batch.token_ids_cpu_tensor.copy_(torch.from_numpy(self.input_batch.token_ids_cpu))
    # Check if the batch has changed. If not, we can skip copying the
    # sampling metadata from CPU to GPU.
    batch_changed = len(removed_req_indices) > 0 or len(req_ids_to_add) > 0

    # Add the new or resumed requests to the persistent batch.
    # The smaller empty indices are filled first.
    removed_req_indices = sorted(removed_req_indices, reverse=True)
    for req_id in req_ids_to_add:
        req_state = self.requests[req_id]
        if removed_req_indices:
            # Fill the empty index.
            req_index = removed_req_indices.pop()
        else:
            # Append to the end.
            req_index = None
        self.input_batch.add_request(req_state, req_index)

    # Condense the batched states if there are empty indices.
    if removed_req_indices:
        self.input_batch.condense(removed_req_indices)

    if batch_changed:
        self.input_batch.refresh_sampling_metadata()


def wrapper_gpu_model_runner_execute_model(func):

    def new_func(*args, **kwargs):
        self = args[0]
        try:
            output = func(*args, **kwargs)
            return output
        except Exception as e:
            logger.warning(
                f"Caught exception {str(e)} when processing req_ids {self.input_batch.req_ids}"
            )
            return ModelRunnerOutput(
                req_ids=self.input_batch.req_ids,
                req_id_to_index=self.input_batch.req_id_to_index,
                sampled_token_ids=None,
                spec_token_ids=None,
                logprobs=None,
                prompt_logprobs_dict={},
            )

    return new_func
