# SPDX-License-Identifier: Apache-2.0

# Functions are adapted from
# https://github.com/vllm-project/vllm/blob/v0.11.0/vllm/v1/spec_decode/eagle.py
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

import mindspore as ms
import numpy as np
import torch
import torch.nn as nn
from vllm.config import get_layers_from_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model
from vllm.model_executor.models import supports_multimodal
from vllm.model_executor.models.deepseek_v2 import DeepseekV32IndexerCache
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch

from vllm_mindspore.model_executor.models.model_base import AttentionWrapper
from vllm_mindspore.v1.attention.backends.ms_attn import (
    MsCommonAttentionMetadata)

logger = init_logger(__name__)


def load_model(self, target_model: nn.Module) -> None:
    # MS uses `AttentionWrapper` instead of `Attention` to manage kv_cache.
    # For mf backend, the model_type is used to distinguish whether it is
    # a mtp model, so that `vllm_config` should be updated.
    # And do not share embed_tokens with target model,
    # because the model struct is different.
    # vllm-mindspore begin
    draft_model_config = \
        self.vllm_config.speculative_config.draft_model_config
    target_attn_layer_names = set(
        get_layers_from_vllm_config(self.vllm_config, AttentionWrapper).keys())
    target_indexer_layer_names = set(
        get_layers_from_vllm_config(self.vllm_config,
                                    DeepseekV32IndexerCache).keys())

    # weight of draft model is not quantized,
    # update model config and quantization config for draft model
    draft_model_config.quantization = None
    target_model_config = self.vllm_config.model_config
    target_quant_config = self.vllm_config.quant_config
    target_cache_dtype = self.vllm_config.cache_config.cache_dtype
    self.vllm_config.cache_config.cache_dtype = "auto"
    self.vllm_config.model_config = draft_model_config
    self.vllm_config.quant_config = None
    self.model = get_model(vllm_config=self.vllm_config,
                           model_config=draft_model_config)
    self.vllm_config.model_config = target_model_config
    self.vllm_config.quant_config = target_quant_config
    self.vllm_config.cache_config.cache_dtype = target_cache_dtype

    draft_attn_layer_names = (get_layers_from_vllm_config(
        self.vllm_config, AttentionWrapper).keys() - target_attn_layer_names)
    # vllm-mindspore end
    indexer_layers = get_layers_from_vllm_config(self.vllm_config,
                                                 DeepseekV32IndexerCache)
    draft_indexer_layer_names = (indexer_layers.keys() -
                                 target_indexer_layer_names)
    self.attn_layer_names = list(draft_attn_layer_names)
    self.indexer_layer_names = list(draft_indexer_layer_names)

    if self.indexer_layer_names:
        first_layer = self.indexer_layer_names[0]
        self.draft_indexer_metadata_builder = (
            indexer_layers[first_layer].get_attn_backend().get_builder_cls()(
                indexer_layers[first_layer].get_kv_cache_spec(),
                self.indexer_layer_names,
                self.vllm_config,
                self.device,
            ))
    else:
        self.draft_indexer_metadata_builder = None

    if supports_multimodal(target_model):
        # handle multimodality
        self.model.config.image_token_index = (
            target_model.config.image_token_index)
        target_language_model = target_model.get_language_model()
    else:
        target_language_model = target_model

    # share lm_head with the target model if needed
    # some model definition do not define lm_head explicitly
    # and reuse embed_tokens for lm_head, e.g., CohereForCausalLM
    if self.vllm_config.speculative_config.method != "eagle3":
        if hasattr(target_language_model, "lm_head"):
            logger.info("Loading EAGLE LM head weights from the target model.")
            self.model.lm_head = target_language_model.lm_head
    else:
        if (hasattr(self.model, "lm_head")
                and hasattr(target_language_model, "lm_head")
                and self.model.lm_head.weight.shape
                == target_language_model.lm_head.weight.shape):
            logger.info("Assuming the EAGLE head shares the same lm_head"
                        " with the target model.")
            del self.model.lm_head
            self.model.lm_head = target_language_model.lm_head
        else:
            logger.info("The EAGLE head's lm_head will be loaded separately"
                        " from the target model.")


def prepare_next_token_ids_padded(
    self,
    common_attn_metadata: CommonAttentionMetadata,
    sampled_token_ids: torch.Tensor,
    requests: dict[str, CachedRequestState],
    gpu_input_batch: InputBatch,
    discard_request_indices: torch.Tensor,
    num_discarded_requests: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    This function is used to prepare the inputs for speculative decoding.
    It calculates the next token ids and the number of valid sampled tokens
    for each request, considering the "discarded" requests whose next token
    is not sampled and comes from `request.get_token_id()` instead.
    It also accounts for the rejected tokens in `sampled_token_ids`.
    This function must use device functions to operate on the inputs, and
    should not introduce any blocking CPU-GPU synchronization.
    """
    # TODO(Ben): Combine this into a custom fused kernel

    # Precompute get_token_id for when there is no valid next token
    # vllm-mindspore begin:
    # use np.array instead of tensor to optimize performance
    num_reqs = gpu_input_batch.num_reqs
    self.backup_next_token_ids.np[:num_reqs] = np.array([
        requests[gpu_input_batch.req_ids[i]].get_token_id(
            common_attn_metadata.seq_lens_np[i].item())
        for i in range(num_reqs)
    ])
    # vllm-mindspore end
    self.backup_next_token_ids.copy_to_gpu(num_reqs)

    # Mask out the sampled tokens indices that should not be sampled.
    discard_sampled_tokens_req_indices = \
        discard_request_indices[:num_discarded_requests]

    valid_sampled_token_ids_gpu = sampled_token_ids.clone()
    valid_sampled_token_ids_gpu.index_fill_(
        0, discard_sampled_tokens_req_indices, -1)

    # Generate a mask for all valid tokens within those requests
    max_gen_len = sampled_token_ids.shape[-1]
    if max_gen_len == 1:
        valid_mask = torch.ones_like(valid_sampled_token_ids_gpu,
                                     dtype=torch.bool)
    else:
        valid_mask = (
            (valid_sampled_token_ids_gpu != -1) &
            (valid_sampled_token_ids_gpu < gpu_input_batch.vocab_size))

    # Count the number of valid tokens in each request
    valid_sampled_tokens_count = valid_mask.sum(dim=1)

    # Get the rightmost valid index per row
    last_valid_indices = valid_sampled_tokens_count - 1
    last_valid_indices_safe = torch.clamp(last_valid_indices, min=0)

    # Get last valid token from each row
    # (assume undefined state where there is no valid token)
    selected_tokens = torch.gather(
        valid_sampled_token_ids_gpu, 1,
        last_valid_indices_safe.unsqueeze(1)).squeeze(1)

    # Use last token if valid, pre-computed backup if not
    batch_size = valid_sampled_token_ids_gpu.shape[0]
    next_token_ids = torch.where(last_valid_indices != -1, selected_tokens,
                                 self.backup_next_token_ids.gpu[:batch_size])

    return next_token_ids, valid_sampled_tokens_count


def prepare_inputs_padded(
    self,
    common_attn_metadata: CommonAttentionMetadata,
    spec_decode_metadata: SpecDecodeMetadata,
    valid_sampled_tokens_count: torch.Tensor,
) -> tuple[CommonAttentionMetadata, torch.Tensor, torch.Tensor]:
    """
    This function is used to prepare the inputs for speculative decoding
    It updates the common_attn_metadata for speculative decoding,
    but does not consider the rejected tokens. Instead, all tokens
    are included as inputs to the speculator, with the rejected tokens
    used as padding and filtered out later by `token_indices_to_sample`.
    No blocking CPU operations should be introduced in this function.
    """
    # `q_seq_lens_np`, `seq_lens_np`, `num_computed_tokens_np` and
    # `slot_mapping_np` is needed for MS-backend to prepare inputs.
    num_draft_tokens_gpu = torch.cat([
        spec_decode_metadata.cu_num_draft_tokens[0:1],
        spec_decode_metadata.cu_num_draft_tokens[1:] -
        spec_decode_metadata.cu_num_draft_tokens[:-1]
    ])

    num_rejected_tokens_gpu = torch.where(
        num_draft_tokens_gpu > 0,
        num_draft_tokens_gpu + 1 - valid_sampled_tokens_count,
        torch.zeros_like(num_draft_tokens_gpu))

    # vllm-mindspore begin:
    # use np.array instead of tensor to optimize performance
    query_start_loc_np = common_attn_metadata.query_start_loc_np

    new_query_len_per_req = (query_start_loc_np[1:] - query_start_loc_np[:-1])

    total_num_tokens = query_start_loc_np[-1].item()
    token_indices = self.token_arange_np[:total_num_tokens]

    spec_common_attn_metadata = MsCommonAttentionMetadata(
        query_start_loc=common_attn_metadata.query_start_loc,
        query_start_loc_cpu=None,
        query_start_loc_np=query_start_loc_np,
        q_seq_lens_np=common_attn_metadata.q_seq_lens_np,
        seq_lens=common_attn_metadata.seq_lens,
        seq_lens_cpu=None,
        seq_lens_np=common_attn_metadata.seq_lens_np,
        num_computed_tokens_cpu=None,
        num_computed_tokens_np=common_attn_metadata.num_computed_tokens_np,
        num_reqs=common_attn_metadata.num_reqs,
        num_actual_tokens=total_num_tokens,
        max_query_len=new_query_len_per_req.max().item(),
        max_seq_len=common_attn_metadata.seq_lens_np.max().item(),
        block_table_tensor=common_attn_metadata.block_table_tensor,
        slot_mapping=None,
        slot_mapping_np=common_attn_metadata.slot_mapping_np[token_indices],
        causal=True,
    )
    # vllm-mindspore end

    token_indices_to_sample = common_attn_metadata.query_start_loc_np[1:] - 1 \
        - num_rejected_tokens_gpu

    return spec_common_attn_metadata, token_indices, token_indices_to_sample


def prepare_inputs(
    self,
    common_attn_metadata: CommonAttentionMetadata,
    sampled_token_ids: list[list[int]],
    num_draft_tokens: list[int],
) -> tuple[CommonAttentionMetadata, torch.Tensor]:
    """
    This function is used to prepare the inputs for speculative decoding.
    It updates to the common_attn_metadata to account for the rejected
    tokens (and newly sampled tokens). It also returns the token indices
    of the tokens that should be fed to the speculator.
    """
    # E.g.
    #  common_attn_metadata.query_start_loc{_cpu}:
    #       [0, q1, q1 + q2, q1 + q2 + q3]                # noqa: ERA001
    #  common_attn_metadata.seq_lens{_cpu}: [s1, s2, s3]
    #  num_rejected_tokens: [n1, n2, n3]                  # noqa: ERA001
    # This function computes the intermediate values:
    #  num_tokens_per_req: [q1 - n1, q2 - n2, q3 - n3]    # noqa: ERA001
    # And returns:
    #  common_attn_metadata.query_start_loc{_cpu}:
    #      [0, q1 - n1, q1 + q2 - n1 - n2,                # noqa: ERA001
    #      q1 + q2 + q3 - n1 - n2 - n3]                   # noqa: ERA001
    #  common_attn_metadata.seq_lens{_cpu}:
    #      [s1 - n1 + 1, s2 - n2 + 1, s3 - n3 + 1]        # noqa: ERA001
    #  token_indices: [0, 1, ..., q1 - n1 - 1,
    #                 q1, q1 + 1, ..., q1 + q2 - n2 - 1,  # noqa: ERA001
    #                 q1 + q2, q1 + q2 + 1, ...,          # noqa: ERA001
    #                 q1 + q2 + q3 - n3 - 1]              # noqa: ERA001

    # `q_seq_lens_np`, `seq_lens_np`, `num_computed_tokens_np` and
    # `slot_mapping_np` is needed for MS-backend to prepare inputs.
    num_rejected_tokens = [
        n + 1 - len(sampled_token_ids[i]) if n > 0 else 0
        for i, n in enumerate(num_draft_tokens)
    ]
    # vllm-mindspore begin: use numpy instead of ms.tensor
    num_rejected_tokens = np.array(num_rejected_tokens, dtype=np.int32)
    query_start_loc_np = common_attn_metadata.query_start_loc_np
    new_seq_lens_np = common_attn_metadata.seq_lens_np - num_rejected_tokens

    # [0, q1, q1 + q2, q1 + q2 + q3] -> [q1, q2, q3]  # noqa: ERA001
    new_query_len_per_req = query_start_loc_np[1:] - query_start_loc_np[:-1]
    # [q1, q2, q3] -> [q1 - n1, q2 - n2, q3 - n3]
    new_num_tokens_per_req_np = new_query_len_per_req - num_rejected_tokens

    # [q1 - n1, q2 - n2, q3 - n3] ->   # noqa: ERA001
    # [0, q1 - n1, q1 + q2 - n1 - n2,  # noqa: ERA001
    #  q1 + q2 + q3 - n1 - n2 - n3]    # noqa: ERA001
    new_query_start_loc_np = np.zeros(query_start_loc_np.shape, dtype=np.int32)
    np.cumsum(new_num_tokens_per_req_np, out=new_query_start_loc_np[1:])

    total_num_tokens = new_query_start_loc_np[-1]
    # Example assuming num_tokens_per_req_np = [2, 4, 3]
    # this implies that `new_query_start_locs` is:
    # [0, 2, 6, 9] ->              # noqa: ERA001
    # [0, 0, 2, 2, 2, 2, 6, 6, 6]  # noqa: ERA001
    #  _r1_  ____r2____  ___r3__
    new_query_start_locs_expanded = np.repeat(new_query_start_loc_np[:-1],
                                              new_num_tokens_per_req_np)
    # [0, 1, 2, 3, 4, 5, 6, 7, 8] ->  # noqa: ERA001
    # [0, 1, 0, 1, 2, 3, 0, 1, 2]     # noqa: ERA001
    #  _r1_  ____r2____  ___r3__
    token_offests = self.token_arange_np[:total_num_tokens] \
        - new_query_start_locs_expanded

    # Expand starting positions to match token pattern
    # [0, q1, q1 + q2] ->                                # noqa: ERA001
    # [0, 0, q1, q1, q1, q1, q1 + q2, q1 + q2, q1 + q2]  # noqa: ERA001
    #  _r1_  _____r2_______  ___________r3____________
    old_query_start_locs_expanded = np.repeat(query_start_loc_np[:-1].numpy(),
                                              new_num_tokens_per_req_np)
    # Final token indices are:
    # [0, 1,                                // req 1    # noqa: ERA001
    #  q1 + 0, q1 + 1, q1 + 2, q1 + 3,       // req 2   # noqa: ERA001
    #  q1 + q2 + 0, q1 + q2 + 1, q1 + q2 + 2] // req 3  # noqa: ERA001
    token_indices = token_offests + old_query_start_locs_expanded

    spec_common_attn_metadata = MsCommonAttentionMetadata(
        query_start_loc=ms.from_numpy(new_query_start_loc_np),
        query_start_loc_cpu=None,
        query_start_loc_np=new_query_start_loc_np,
        seq_lens_cpu=None,
        seq_lens_np=new_seq_lens_np,
        seq_lens=ms.from_numpy(new_seq_lens_np),
        q_seq_lens_np=new_num_tokens_per_req_np,
        num_computed_tokens_cpu=None,
        num_computed_tokens_np=common_attn_metadata.num_computed_tokens_np,
        num_reqs=common_attn_metadata.num_reqs,
        num_actual_tokens=total_num_tokens,
        max_query_len=new_query_len_per_req.max().item(),
        max_seq_len=new_seq_lens_np.max().item(),
        block_table_tensor=common_attn_metadata.block_table_tensor,
        slot_mapping=None,
        slot_mapping_np=common_attn_metadata.slot_mapping_np[token_indices],
        causal=True,
    )
    # vllm-mindspore end
    return spec_common_attn_metadata, token_indices
