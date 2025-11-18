# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2025 Huawei Technologies Co., Ltd.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Functions are adapted from
# https://github.com/vllm-project/vllm/blob/v0.9.1/vllm/v1/engine/processor.py

from vllm.sampling_params import SamplingParams


def v1_process_validate_sampling_params(
    self,
    params: SamplingParams,
) -> None:

    model_config = self.vllm_config.model_config
    vocab_size = model_config.get_vocab_size()
    if params.top_k > vocab_size:
        raise ValueError(
            f"top_k cannot be greater than vocabulary size({vocab_size}), "
            f"but got {params.top_k}.")
    scheduler_config = self.vllm_config.scheduler_config
    max_num_seqs = scheduler_config.max_num_seqs
    if params.n > max_num_seqs:
        raise ValueError(f"SchedulerConfig.n cannot be greater than "
                         f"max_num_seqs({max_num_seqs}), but got {params.n}.")

    self._validate_structured_output(params)
    self._validate_logit_bias(params)

    if params.allowed_token_ids is None:
        return
    if not params.allowed_token_ids:
        raise ValueError("allowed_token_ids is not None and empty!")
    vocab_size = len(self.tokenizer)
    if not all(0 <= tid < vocab_size for tid in params.allowed_token_ids):
        raise ValueError("allowed_token_ids contains out-of-vocab token id!")


def v1_process_validate_structured_output(self,
                                          params: SamplingParams) -> None:
    if not params.guided_decoding or not self.decoding_config:
        return
    raise ValueError(
        "vLLM-MindSpore Plugin does not support structured output now.")
