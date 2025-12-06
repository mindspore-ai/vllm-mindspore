# SPDX-License-Identifier: Apache-2.0

# Adapted from
# https://github.com/vllm-project/vllm/blob/main/tests/v1/core/
# test_scheduler.py
#
# Copyright 2025 Huawei Technologies Co., Ltd.
# Copyright 2024-2025 The vLLM team.
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
"""Test EnhancedScheduler."""
import os
from typing import Optional
import pytest
from unittest.mock import patch

import vllm_mindspore
import mindspore as ms

from vllm.config import (CacheConfig, KVTransferConfig, ModelConfig,
                         SchedulerConfig, SpeculativeConfig, VllmConfig)
from vllm.multimodal.inputs import MultiModalKwargs, PlaceholderRange
from vllm.utils import sha256
from vllm.v1.core.kv_cache_utils import (get_request_block_hasher,
                                         init_none_hash)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheConfig,
                                        KVCacheGroupSpec, KVCacheTensor)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager
from vllm import SamplingParams

from vllm_mindspore.v1.core.sched.scheduler import EnhancedScheduler

from tests.utils.common_utils import (teardown_function, setup_function,
                                      MODEL_PATH)

EOS_TOKEN_ID = 151645
QWEN3_MODEL_PATH = MODEL_PATH["Qwen3-0.6B"]


def create_scheduler(
    model: str = QWEN3_MODEL_PATH,
    max_num_seqs: int = 16,
    max_num_batched_tokens: int = 8192,
    enable_chunked_prefill: bool = True,
    enable_prefix_caching: bool = True,
    long_prefill_token_threshold: int = 0,
    disable_chunked_mm_input: bool = False,
    num_blocks: int = 10000,
    block_size: int = 16,
    max_model_len: Optional[int] = None,
):
    if max_model_len is None:
        max_model_len = max_num_batched_tokens
    scheduler_config = SchedulerConfig(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        max_model_len=max_model_len,
        long_prefill_token_threshold=long_prefill_token_threshold,
        disable_chunked_mm_input=disable_chunked_mm_input,
        enable_chunked_prefill=enable_chunked_prefill,
    )
    model_config = ModelConfig(
        model=model,
        task="auto",
        tokenizer=model,
        tokenizer_mode="auto",
        trust_remote_code=True,
        dtype="float16",
        seed=42,
    )
    # Cache config, optionally force APC
    kwargs_cache = ({} if enable_prefix_caching is None else {
        'enable_prefix_caching': enable_prefix_caching
    })
    cache_config = CacheConfig(
        block_size=block_size,
        gpu_memory_utilization=0.9,
        swap_space=0,
        cache_dtype="auto",
        **kwargs_cache,
    )
    vllm_config = VllmConfig(
        scheduler_config=scheduler_config,
        model_config=model_config,
        cache_config=cache_config,
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,  # A large number of blocks to hold all requests
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(['layer'],
                             FullAttentionSpec(block_size, 1, 1, ms.float32,
                                               False))
        ],
    )
    cache_config.num_gpu_blocks = num_blocks
    return EnhancedScheduler(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(vllm_config),
    )


_none_hash_initialized = False


def create_requests(
    num_requests: int,
    num_tokens: int = 10,
    mm_positions: Optional[list[list[PlaceholderRange]]] = None,
    max_tokens: int = 16,
    stop_token_ids: Optional[list[int]] = None,
    prompt_logprobs: Optional[int] = None,
    same_prompt: bool = False,
    block_size: int = 16,
) -> list[Request]:
    global _none_hash_initialized
    if not _none_hash_initialized:
        init_none_hash(sha256)
        _none_hash_initialized = True

    block_hasher = get_request_block_hasher(block_size, sha256)
    sampling_params = SamplingParams(ignore_eos=False,
                                     max_tokens=max_tokens,
                                     stop_token_ids=stop_token_ids,
                                     prompt_logprobs=prompt_logprobs)
    requests = []
    for i in range(num_requests):
        prompt_token_ids = ([0] * num_tokens if same_prompt else [i] *
                            num_tokens)
        request = Request(
            request_id=f"{i}",
            prompt_token_ids=prompt_token_ids,
            sampling_params=sampling_params,
            pooling_params=None,
            mm_features=None,
            eos_token_id=EOS_TOKEN_ID,
            block_hasher=block_hasher,
        )
        requests.append(request)
    return requests


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_chunked_prefixcache_schedule():
    """
    Test Summary:
        默认开启chunked_prefill+prefixcache时, 测试schedule接口的功能
    Expected Result:
        默认调度策略, 不区分prefill/decode
    Model Info:
        Qwen3-0.6B
    """
    scheduler = create_scheduler(max_num_batched_tokens=1024,
                                 enable_prefix_caching=True,
                                 enable_chunked_prefill=True)
    # 15 * 1000 > 1024
    requests = create_requests(num_requests=15, num_tokens=100)
    for request in requests:
        scheduler.add_request(request)

    # Test initial scheduling
    output = scheduler.schedule()
    assert len(output.scheduled_new_reqs) == 11
    assert output.scheduled_cached_reqs.num_reqs == 0
    assert len(output.finished_req_ids) == 0

    # Verify part requests are scheduled.
    for req_id, num_tokens in output.num_scheduled_tokens.items():
        assert (num_tokens == len(requests[int(req_id)].prompt_token_ids)
                or num_tokens == 24)

    # Verify requests moved from waiting to running
    assert len(scheduler.waiting) == 4
    assert len(scheduler.running) == 11
    for i, request in enumerate(requests):
        if i < len(scheduler.running):
            assert scheduler.running[i] == request
        else:
            assert request in scheduler.waiting

    # second scheduling
    output = scheduler.schedule()
    assert len(output.scheduled_new_reqs) == 4
    assert output.scheduled_cached_reqs.num_reqs == 1
    assert len(output.finished_req_ids) == 0
    assert len(scheduler.waiting) == 0
    assert len(scheduler.running) == 15


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_nochunk_noprefixcache_schedule():
    """
    Test Summary:
        关闭chunked_prefill和prefixcache, 测试schedule接口的功能
    Expected Result:
        prefill-first策略调度
    Model Info:
        Qwen3-0.6B
    """
    scheduler = create_scheduler(max_num_batched_tokens=1024,
                                 enable_prefix_caching=False,
                                 enable_chunked_prefill=False)
    # 15 * 1000 > 1024
    requests = create_requests(num_requests=15, num_tokens=100)
    for request in requests:
        scheduler.add_request(request)

    # Test initial scheduling
    output = scheduler.schedule()
    assert len(output.scheduled_new_reqs) == 10
    assert output.scheduled_cached_reqs.num_reqs == 0
    assert len(output.finished_req_ids) == 0

    # Verify part requests are scheduled.
    for req_id, num_tokens in output.num_scheduled_tokens.items():
        assert num_tokens == len(requests[int(req_id)].prompt_token_ids)

    # Verify requests moved from waiting to running
    assert len(scheduler.waiting) == 5
    assert len(scheduler.running) == 10
    for i, request in enumerate(requests):
        if i < len(scheduler.running):
            assert scheduler.running[i] == request
        else:
            assert request in scheduler.waiting

    # second scheduling
    output = scheduler.schedule()
    assert len(output.scheduled_new_reqs) == 5
    assert output.scheduled_cached_reqs.num_reqs == 0
    assert len(output.finished_req_ids) == 0
    assert len(scheduler.waiting) == 0
    assert len(scheduler.running) == 15


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_nochunked_prefixcache_schedule():
    """
    Test Summary:
        关闭chunked_prefill时(prefixcache默认打开),测试schedule接口的功能
    Expected Result:
        无重复请求时, 会按prefill-first策略调度;
        有重复请求时, 新请求命中prefixcache走decode
    Model Info:
        Qwen3-0.6B
    """
    scheduler = create_scheduler(max_num_batched_tokens=1024,
                                 enable_prefix_caching=True,
                                 enable_chunked_prefill=False)

    # create same requests: 15 * 100 > 1024
    requests = create_requests(num_requests=15,
                               num_tokens=100,
                               same_prompt=True)
    for request in requests:
        scheduler.add_request(request)

    # Test initial scheduling
    output = scheduler.schedule()

    # All requests have been scheduled because of prefixcache.
    assert len(output.scheduled_new_reqs) == 15
    assert output.scheduled_cached_reqs.num_reqs == 0
    assert len(output.finished_req_ids) == 0

    # Verify requests moved from waiting to running
    assert len(scheduler.waiting) == 0
    assert len(scheduler.running) == 15
