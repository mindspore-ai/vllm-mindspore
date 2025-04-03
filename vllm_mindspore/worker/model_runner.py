#!/usr/bin/env python3
# encoding: utf-8
# Copyright 2025 Huawei Technologies Co., Ltd
# Copyright 2024 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

from typing import List

import torch
from vllm.distributed import get_pp_group
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.sampling_params import SamplingParams
from vllm.sequence import SequenceGroupMetadata
from vllm_mindspore.utils import STR_DTYPE_TO_TENSOR_DTYPE

from mindspore import mutable

logger = init_logger(__name__)

LORA_WARMUP_RANK = 8


def _get_cuda_graph_pad_size(
    self, num_seqs: int, max_decode_seq_len: int, max_encoder_seq_len: int = 0
) -> int:
    # No need to use cuda graph for mindspore.
    return -1


def _dummy_run(self,
               max_num_batched_tokens: int,
               max_num_seqs: int = 1) -> None:
    with self.set_in_profile_run():
        # Enable top-k sampling to reflect the accurate memory usage.
        sampling_params = \
            SamplingParams(top_p=0.99, top_k=self.vocab_size - 1)

        # This represents the maximum number of different requests
        # that will have unique loras, an therefore the max amount of memory
        # consumption create dummy lora request copies from the lora request
        # passed in, which contains a lora from the lora warmup path.
        dummy_lora_requests: List[LoRARequest] = []
        dummy_lora_requests_per_seq: List[LoRARequest] = []
        if self.lora_config:
            assert self.lora_manager is not None
            with self.lora_manager.dummy_lora_cache():
                for idx in range(self.lora_config.max_loras):
                    lora_id = idx + 1
                    dummy_lora_request = LoRARequest(
                        lora_name=f"warmup_{lora_id}",
                        lora_int_id=lora_id,
                        lora_path="/not/a/real/path",
                    )
                    self.lora_manager.add_dummy_lora(dummy_lora_request,
                                                        rank=LORA_WARMUP_RANK)
                    dummy_lora_requests.append(dummy_lora_request)
                dummy_lora_requests_per_seq = [
                    dummy_lora_requests[idx % len(dummy_lora_requests)]
                    for idx in range(max_num_seqs)
                ]

        # Profile memory usage with max_num_sequences sequences and the
        # total number of tokens equal to max_num_batched_tokens.
        seqs: List[SequenceGroupMetadata] = []
        # Additional GPU memory may be needed for multi-modal encoding,
        # which needs to be accounted for when calculating the GPU blocks
        # for vLLM blocker manager.
        # To exercise the worst scenario for GPU memory consumption,
        # the number of seqs (batch_size) is chosen to maximize the number
        # of images processed.

        max_mm_tokens = self.mm_registry.get_max_multimodal_tokens(
            self.model_config)
        if max_mm_tokens > 0:
            max_num_seqs_orig = max_num_seqs
            max_num_seqs = min(max_num_seqs,
                                max_num_batched_tokens // max_mm_tokens)
            if max_num_seqs < 1:
                expr = (f"min({max_num_seqs_orig}, "
                        f"{max_num_batched_tokens} // {max_mm_tokens})")
                logger.warning(
                    "Computed max_num_seqs (%s) to be less than 1. "
                    "Setting it to the minimum value of 1.", expr)
                max_num_seqs = 1

        batch_size = 0
        for group_id in range(max_num_seqs):
            seq_len = (max_num_batched_tokens // max_num_seqs +
                        (group_id < max_num_batched_tokens % max_num_seqs))
            batch_size += seq_len

            dummy_data = self.input_registry \
                .dummy_data_for_profiling(self.model_config,
                                        seq_len,
                                        self.mm_registry)

            seq = SequenceGroupMetadata(
                request_id=str(group_id),
                is_prompt=True,
                seq_data={group_id: dummy_data.seq_data},
                sampling_params=sampling_params,
                block_tables=None,
                lora_request=dummy_lora_requests_per_seq[group_id]
                if dummy_lora_requests_per_seq else None,
                multi_modal_data=dummy_data.multi_modal_data,
                multi_modal_placeholders=dummy_data.
                multi_modal_placeholders,
            )
            seqs.append(seq)

        # Run the model with the dummy inputs.
        num_layers = self.model_config.get_num_layers(self.parallel_config)
        # use an empty tensor instead of `None`` to force Dynamo to pass
        # it by reference, rather by specializing on the value ``None``.
        # the `dtype` argument does not matter, and we use `float32` as
        # a placeholder (it has wide hardware support).
        # it is important to create tensors inside the loop, rather than
        # multiplying the list, to avoid Dynamo from treating them as
        # tensor aliasing.
        kv_cache_dtype = self.model_config.dtype if self.cache_config.cache_dtype == "auto" \
            else self.cache_config.cache_dtype
        kv_cache_dtype = STR_DTYPE_TO_TENSOR_DTYPE[kv_cache_dtype]
        block_size = self.cache_config.block_size
        num_kv_heads = self.model_config.get_num_kv_heads(self.parallel_config)
        head_size = self.model_config.get_head_size()
        kv_shape = [0, block_size, num_kv_heads, head_size]
        kv_caches = mutable([
            mutable((
                mutable(torch.tensor([], dtype=kv_cache_dtype, device=self.device).reshape(kv_shape)),
                mutable(torch.tensor([], dtype=kv_cache_dtype, device=self.device).reshape(kv_shape)),
            ))
            for _ in range(num_layers)
        ])
        finished_requests_ids = [seq.request_id for seq in seqs]
        model_input = self.prepare_model_input(
            seqs, finished_requests_ids=finished_requests_ids)
        intermediate_tensors = None
        if not get_pp_group().is_first_rank:
            intermediate_tensors = \
                self.model.make_empty_intermediate_tensors(
                batch_size=batch_size,
                dtype=self.model_config.dtype,
                device=self.device)

        # Disable KV Scale Calculation for dummy data during profile run
        if model_input.attn_metadata is not None:
            model_input.attn_metadata.enable_kv_scales_calculation = False

        self.execute_model(model_input, kv_caches, intermediate_tensors)
        torch.cuda.synchronize()
        if self.lora_config:
            # Remove dummy loras.
            assert self.lora_manager is not None
            self.remove_all_loras()
        return


MULTI_STEP_ATTENTION_BACKENDS = [
    "MS_MLA", "MS_ATTN", "NO_ATTENTION"
]
MULTI_STEP_CHUNKED_PREFILL_ATTENTION_BACKENDS = ["MS_MLA", "MS_ATTN"]

def _get_supported_attention_backends(chunked_prefill_enabled: bool) \
    -> List[str]:
    if chunked_prefill_enabled:
        return MULTI_STEP_CHUNKED_PREFILL_ATTENTION_BACKENDS
    else:
        return MULTI_STEP_ATTENTION_BACKENDS