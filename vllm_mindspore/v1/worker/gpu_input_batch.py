from typing import Dict, List, Optional, Set, Tuple, cast

import numpy as np
import torch

from vllm.lora.request import LoRARequest
from vllm.sampling_params import SamplingType
from vllm.v1.sample.metadata import SamplingMetadata
from vllm_mindspore.v1.utils import copy_slice
from vllm.v1.worker.block_table import BlockTable

_SAMPLING_EPS = 1e-5


def _make_sampling_metadata(self) -> SamplingMetadata:
    num_reqs = self.num_reqs
    if not self.all_greedy:
        temperature = copy_slice(torch.from_numpy(self.temperature_cpu), self.temperature, num_reqs)
        temperature = temperature[:num_reqs]
    else:
        temperature = None
    if not self.no_top_p:
        copy_slice(torch.from_numpy(self.top_p_cpu), self.top_p, num_reqs)
    if not self.no_top_k:
        copy_slice(torch.from_numpy(self.top_k_cpu), self.top_k, num_reqs)
    if not self.no_min_p:
        copy_slice(torch.from_numpy(self.min_p_cpu), self.min_p, num_reqs)

    if not self.no_penalties:
        # Since syncing these tensors is expensive only copy them
        # if necessary i.e. if there are requests which require
        # penalties to be applied during sampling.
        copy_slice(torch.from_numpy(self.frequency_penalties_cpu),
                self.frequency_penalties, num_reqs)
        copy_slice(torch.from_numpy(self.presence_penalties_cpu),
                self.presence_penalties, num_reqs)
        copy_slice(torch.from_numpy(self.repetition_penalties_cpu),
                self.repetition_penalties, num_reqs)

        # The prompt tokens are used only for applying penalties during
        # the sampling process. Hence copy these tensors only when
        # there are requests which need penalties to be applied.
        prompt_token_ids = self._make_prompt_token_ids_tensor()
    else:
        prompt_token_ids = None

    allowed_token_ids_mask: Optional[torch.Tensor] = None
    if not self.no_allowed_token_ids:
        assert self.allowed_token_ids_mask is not None
        copy_slice(self.allowed_token_ids_mask_cpu_tensor,
                    self.allowed_token_ids_mask, num_reqs)
        allowed_token_ids_mask = self.allowed_token_ids_mask[:num_reqs]

    return SamplingMetadata(
        temperature=temperature,
        all_greedy=self.all_greedy,
        all_random=self.all_random,
        top_p=None if self.no_top_p else self.top_p[:num_reqs],
        top_k=None if self.no_top_k else self.top_k[:num_reqs],
        min_p=None if self.no_min_p else self.min_p[:num_reqs],
        generators=self.generators,
        max_num_logprobs=self.max_num_logprobs,
        prompt_token_ids=prompt_token_ids,
        frequency_penalties=self.frequency_penalties[:num_reqs],
        presence_penalties=self.presence_penalties[:num_reqs],
        repetition_penalties=self.repetition_penalties[:num_reqs],
        output_token_ids=cast(list[list[int]], self.req_output_token_ids),
        min_tokens=self.min_tokens,
        no_penalties=self.no_penalties,
        logit_bias=self.logit_bias[:num_reqs],
        allowed_token_ids_mask=allowed_token_ids_mask,
        bad_words_token_ids=self.bad_words_token_ids,
    )


def _make_prompt_token_ids_tensor(self) -> torch.Tensor:
    max_prompt_len = self.num_prompt_tokens[:self.num_reqs].max()
    prompt_token_ids = np.empty((self.num_reqs, max_prompt_len), dtype=np.int64)
    prompt_token_ids[:] = self.token_ids_cpu[:self.
                                             num_reqs, :max_prompt_len]
    for i in range(self.num_reqs):
        prompt_token_ids[i, self.num_prompt_tokens[i]:] = self.vocab_size
    prompt_token_ids_cpu_tensor = torch.from_numpy(prompt_token_ids)
    prompt_token_ids_cpu_tensor.move_to("Ascend", blocking=False)
    return prompt_token_ids_cpu_tensor

