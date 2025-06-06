# Copyright 2025 Huawei Technologies Co., Ltd
#
# Functions is mainly Adapted from https://github.com/vllm-project/vllm/blob/v0.8.3/vllm/scheduler.py
# Copyright 2025 The vLLM team.
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

import os
import json
import numpy as np
import torch

from typing import Optional, Mapping, List

from mindspore.common._stub_tensor import StubTensor
from torch import _tensor
from vllm.sequence import SequenceGroup, Sequence
from vllm import SamplingParams
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.pooling_params import PoolingParams
from vllm.prompt_adapter.request import PromptAdapterRequest

logger = init_logger(__name__)

original_init = SequenceGroup.__init__

def stub_tensor_numel(self):
    from functools import reduce
    shape = self.shape
    return reduce((lambda x, y: x * y), shape) if shape else 1

def patched_init(
    self,
    request_id: str,
    seqs: List[Sequence],
    arrival_time: float,
    sampling_params: Optional[SamplingParams] = None,
    lora_request: Optional[LoRARequest] = None,
    pooling_params: Optional[PoolingParams] = None,
    pooled_data: Optional[torch.Tensor] = None,
    encoder_seq: Optional[Sequence] = None,
    trace_headers: Optional[Mapping[str, str]] = None,
    prompt_adapter_request: Optional[PromptAdapterRequest] = None,
    priority: int = 0,
    draft_size: int = 1,
) -> None:
    original_init(
        self, request_id, seqs, arrival_time, sampling_params,
        lora_request, pooling_params, pooled_data, encoder_seq,
        trace_headers, prompt_adapter_request, priority, draft_size
    )
    
    # vllm-mindspore begin:
    # Add params to get optimized chunk-sizes
    default_paras = [3.18223033e+02, -3.59852154e-01, 1.15576435e-04, -2.09428155e-02]
    if 'ADAPTIVE_CHUNK_PARAMS' in os.environ:
        # set your params as:
        # export ADAPTIVE_CHUNK_PARAMS='[302.182, -0.359, 0.000115, -0.02094]'
        # You can get the parameters using the following method:
        # def fit_parameters(q: np.ndarray, pre_kv: np.ndarray, time: np.ndarray) -> np.ndarray:
        #     # get fitting parameters from data
        #     A = np.column_stack([
        #         np.ones_like(q),        # k1
        #         q,                      # k2
        #         q * (q + pre_kv),       # k3 
        #         (q + pre_kv)            # k4
        #     ])
        #     return np.linalg.lstsq(A, time, rcond=None)[0]
        # paras = fit_parameters(q, pre_kv, time)
        try:
            params = json.loads(os.environ['ADAPTIVE_CHUNK_PARAMS'])
            if isinstance(params, list) and len(params) == 4:
                self.params = params
                logger.info(f'Using custom params: {self.params}')
            else:
                raise ValueError
        except (json.JSONDecodeError, ValueError):
            logger.warning('Invalid params format, using defaults')
            self.params = default_paras
    else:
        self.params = default_paras
        logger.info(f'Using default params(Qwen2-7B-Instruct): {self.params}')
    self.chunk_sizes = get_optimize_chunks(len(self.first_seq.prompt_token_ids), self.params)
    self.chunk_index = 0
    logger.info(f'Prompt length:{len(self.first_seq.prompt_token_ids)}')
    logger.info(f'Optimized chunk sizes:{self.chunk_sizes}')
    # vllm-mindspore end.

def calculate_layer_time(params, q, pre_kv_len):
    k1, k2, k3, k4 = params
    len_kv = q + pre_kv_len
    return k1 + k2*q + k3*q*len_kv + k4*len_kv

def optimize_chunks(params, seq_len, chunk_num):
    # Calculate optimal chunk strategy based on sequence length, chunk num, and params
    # Use fixed split average time as the target for optimization  
    avg_len = seq_len / chunk_num
    target_time = np.mean([
        calculate_layer_time(params, avg_len, i*avg_len)
        for i in range(chunk_num)
    ])
    chunks = []
    remaining = seq_len
    pre_kv = 0
    # Compute the length distribution for chunks 0 to n-2
    for _ in range(chunk_num - 1):
        try:
            q = solve_for_q(params, target_time, pre_kv)
            q = max(32, min(q, remaining))
            q = int(round(q / 32) * 32)
            chunks.append(q)
            remaining -= q
            pre_kv += q
        except:
            chunks.append(remaining)
            break
    chunks.append(remaining)
    # Apply adaptive tuning
    for _ in range(5):
        times = [calculate_layer_time(params, q, sum(chunks[:i])) 
                for i, q in enumerate(chunks)]
        max_idx = np.argmax(times)
        min_idx = np.argmin(times)
        if max_idx == min_idx:
            break
        delta = 32 
        if chunks[max_idx] <= delta:
            break
        new_chunks = chunks.copy()
        new_chunks[max_idx] -= delta
        new_chunks[min_idx] += delta
        if sum(new_chunks) == seq_len and all(q > 0 for q in new_chunks):
            chunks = new_chunks
    return chunks

def solve_for_q( params, target_time, pre_kv_len):
    k1, k2, k3, k4 = params
    a = k3
    b = k2 + k3*pre_kv_len + k4
    c = k1 + k4*pre_kv_len - target_time
    discriminant = b**2 - 4*a*c
    if discriminant < 0:
        raise ValueError("No real solution")
    q = max((-b + np.sqrt(discriminant))/(2*a),
            (-b - np.sqrt(discriminant))/(2*a))
    return max(0, q)

def get_optimize_chunks(length, params):
    chunk_num = (length-1) // 2048 + 1
    return [length] if chunk_num == 1 else optimize_chunks(params, length, chunk_num)

def get_next_chunk_size(self) -> Optional[int]:
    chunk_size = self.chunk_sizes[self.chunk_index]
    self.chunk_index += 1
    return chunk_size

def apply_sequence_patch():
    _tensor.stub_tensor_numel=stub_tensor_numel
    setattr(StubTensor, 'numel', stub_tensor_numel)
    if 'ADAPTIVE_CHUNK' in os.environ and os.environ['ADAPTIVE_CHUNK'] == '1':
        SequenceGroup.__init__ = patched_init
        SequenceGroup.get_next_chunk_size = get_next_chunk_size
