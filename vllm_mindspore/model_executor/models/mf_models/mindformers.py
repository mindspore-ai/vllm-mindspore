# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 Huawei Technologies Co., Ltd.
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
import os
from collections.abc import Iterable
from typing import Optional, Union

import mindspore as ms
import numpy as np
from mindformers import AutoModel, PreTrainedModel
from mindformers.core.context import build_mf_context
from mindformers.tools.utils import is_pynative
from mindspore import Tensor, mutable, ops
from mindspore.nn.utils import no_init_parameters
from vllm import envs
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed.parallel_state import get_dp_group, get_pp_group
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import SupportsPP
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors

from vllm_mindspore.model_executor.models.attention_mask import (
    LowerTriangularMask, MLALowerTriangularMask)
from vllm_mindspore.model_executor.models.mf_models.config import gen_mf_config
from vllm_mindspore.model_executor.models.model_base import (
    AttentionWrapper, MLAAttentionWrapper, MsModelBase)
from vllm_mindspore.model_executor.models.utils import (
    is_use_ringmla, make_empty_intermediate_tensors_factory)
from vllm_mindspore.utils import is_310p

logger = init_logger(__name__)


class MindFormersForCausalLM(MsModelBase, SupportsPP):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.set_flags = False
        self.model_config = vllm_config.model_config
        self.lm_head_graph = None

        mf_config = gen_mf_config(vllm_config)
        mf_config.load_checkpoint = self.get_model_path()
        mf_config.pretrained_model_dir = self.get_model_path()
        self.mf_config = mf_config
        self.mla_config = self.mf_config.get('model', None).get(
            'model_config', None).get('multi_latent_attention', False)
        self.use_ringmla = is_use_ringmla(vllm_config, mf_config)
        self.is_chunked = False

        build_mf_context(self.mf_config)

        self.network, self.lm_head = self._create_network()
        self.casual_mask = self._create_mask()

        self._set_dynamic_inputs()

        self.set_modules({"model": self.network})

        num_layers = self.model_config.get_num_layers(self.parallel_config)
        self.kv_caches = self._create_kv_caches(num_layers)
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        for i in range(num_layers):
            compilation_config.static_forward_context[str(
                i)] = self.kv_caches[i]

        self.make_empty_intermediate_tensors = \
            make_empty_intermediate_tensors_factory(
                keys=["hidden_states"],
                hidden_size=self.model_config.hf_config.hidden_size)

        self.cast = ops.Cast()

    def _set_dynamic_inputs(self):
        self.network.set_dynamic_inputs()
        dynamic_hidden_states = Tensor(shape=[None, None],
                                       dtype=self.network.compute_dtype)
        if get_pp_group().is_last_rank:
            self.lm_head.set_inputs(dynamic_hidden_states)

    def _create_mask(self):
        # Initial mask
        mask_func = (MLALowerTriangularMask
                     if self.mla_config else LowerTriangularMask)
        return mask_func(dtype=self.network.compute_dtype,
                         max_model_len=self.model_config.max_model_len)

    def _create_kv_caches(self, num_layers):
        # Initial kv_caches
        wrapper_func = (MLAAttentionWrapper
                        if self.mla_config else AttentionWrapper)
        return [wrapper_func() for _ in range(num_layers)]

    def get_kvcache(self):
        if not self.mla_config:
            return super().get_kvcache()

        key_cache = []
        forward_context = get_forward_context()
        key_cache = [
            self.kv_caches[i].kv_cache[forward_context.virtual_engine][0]
            for i in range(self.config.num_hidden_layers)
        ]
        if not self.use_ringmla:
            return mutable(key_cache), None
        # deepseek mla op need key cache and rope cache
        rope_cache = [
            self.kv_caches[i].kv_cache[forward_context.virtual_engine][1]
            for i in range(self.config.num_hidden_layers)
        ]
        return mutable(key_cache), mutable(rope_cache)

    def _get_padding_index(self, q_seq_len):
        """
        Calculate the padding index used in the mixed parallel scenario.
        Case 1: When data_parallel_size equals 1, no padding operation
                required, returns None.
        Case 2: When data_parallel_size equals expert_parallel_size and
                model_parallel equals 1, all_to_all communication is applied,
                no padding operation required, returns None.
        Case 3: In other DP enabled scenarios, calculate the corresponding
                padding index based on the query sequence lengths processed
                by each DP domain.

        e.g. DP2 TP4 MoE_EP2
        +------------------+------------------------+------------------------+
        |    DP domain     |          DP0           |           DP1          |
        +------------------+------------------------+------------------------+
        |    q_seq_len     |           3            |            5           |
        +------------------+------------------------+------------------------+
        | attn_padding_idx |   [0,1,2,0,0,0,0,0]    |   [0,1,2,3,4,0,0,0]    |
        +------------------+------------------------+------------------------+
        |attn_unpadding_idx|               [0,1,2,8,9,10,11,12]              |
        +------------------+------------------------+------------------------+
        | ffn_padding_idx  |        [0,1,2,0,0,0,0,0,3,4,5,6,7,0,0,0]        |
        +------------------+------------------------+------------------------+
        |ffn_unpadding_idx |        [0,1,2]         |      [0,1,2,3,4]       |
        +------------------+------------------------+------------------------+

        Args:
        - q_seq_len (Tensor): query sequence lengths.

        Returns:
        - attn_padding_idx (Tensor or None): Indices mapping positions in
          attention output sequence to original token positions, used for
          padding attention output to fixed size.
        - attn_unpadding_idx (Tensor or None): Indices mapping valid tokens
          in padded attention output sequence to their original positions,
          used for removing padding in attention output.
        - ffn_padding_idx (Tensor or None): Indices mapping positions in MoE
          output sequence to flattened valid token positions, used for padding
          MoE output to fixed size.
        - ffn_unpadding_idx (Tensor or None): Indices mapping valid tokens in
          padded MoE output sequence to their original positions, used for
          removing padding in MoE output.
        """
        dp_size = self.mf_config.parallel_config.data_parallel
        tp_size = self.mf_config.parallel_config.model_parallel
        ep_size = self.mf_config.parallel_config.expert_parallel
        if dp_size == 1 or (dp_size == ep_size and tp_size == 1):
            return None, None, None, None

        tokens_len_per_dp = q_seq_len.sum().reshape(-1)
        tokens_len_per_dp = get_dp_group().all_gather(tokens_len_per_dp)
        tokens_len_per_dp = tokens_len_per_dp.asnumpy()

        # Simultaneously satisfying the requirement of being divisible by
        # tensor_parallel_size and greater than the maximum q_seq_len in all
        # DP domains.
        padding_size = ((tokens_len_per_dp.max() + tp_size - 1) // tp_size *
                        tp_size)

        dp_rank_id = get_dp_group().rank_in_group
        attn_padding_idx = None
        attn_unpadding_idx = None
        ffn_padding_idx = None
        ffn_unpadding_idx = None
        last_arange_index = 0

        for dp_rank, tokens_length in enumerate(tokens_len_per_dp):
            arange_data = np.arange(0, int(tokens_length), dtype=np.int32)
            if dp_rank == dp_rank_id:
                ffn_unpadding_idx = arange_data
                pad = np.zeros(padding_size - arange_data.shape[0],
                               dtype=np.int32)
                attn_padding_idx = np.concatenate((arange_data, pad), axis=0)
            if dp_rank == 0:
                attn_unpadding_idx = arange_data
                last_arange_index = arange_data[-1]
                pad = np.zeros(padding_size - attn_unpadding_idx.shape[0],
                               dtype=np.int32)
                ffn_padding_idx = np.concatenate((attn_unpadding_idx, pad),
                                                 axis=0)
            else:
                attn_offset_idx = arange_data + padding_size * dp_rank
                attn_unpadding_idx = np.concatenate(
                    (attn_unpadding_idx, attn_offset_idx), axis=0)
                ffn_offset_idx = arange_data + last_arange_index + 1
                last_arange_index = ffn_offset_idx[-1]
                pad = np.zeros(padding_size - ffn_offset_idx.shape[0],
                               dtype=np.int32)
                ffn_padding_idx = np.concatenate(
                    (ffn_padding_idx, ffn_offset_idx, pad), axis=0)
        return (ms.from_numpy(attn_padding_idx),
                ms.from_numpy(attn_unpadding_idx),
                ms.from_numpy(ffn_padding_idx),
                ms.from_numpy(ffn_unpadding_idx))

    def update_padding_index_to_inputs(self, model_inputs, q_seq_lens):
        """
        Update the model input and add the related parameters of padding_index.
        """

        (attn_padding_idx, attn_unpadding_idx, ffn_padding_idx,
         ffn_unpadding_idx) = self._get_padding_index(q_seq_lens)

        model_inputs["attn_padding_idx"] = attn_padding_idx
        model_inputs["attn_unpadding_idx"] = attn_unpadding_idx
        model_inputs["ffn_padding_idx"] = ffn_padding_idx
        model_inputs["ffn_unpadding_idx"] = ffn_unpadding_idx

        return model_inputs

    def prepare_inputs(self, input_ids, positions):

        attn_metadata = get_forward_context().attn_metadata
        # 0.9.1 attn_metadata[layer_name], don't have layer_name here
        # so we just take one by default
        if isinstance(attn_metadata, dict) and '1' in attn_metadata:
            attn_metadata = attn_metadata['1']
        if attn_metadata is None:
            attn_metadata = self._dummy_attention_metadata(
                input_ids, positions)
        key_cache, value_cache = self.get_kvcache()
        if not envs.VLLM_USE_V1:
            # V0
            seq_lens = attn_metadata.seq_lens
            max_query_len = attn_metadata.max_query_len
            # When Mutli-Step is enabled with Chunked-Prefill, prefills and
            # decodes are scheduled together. In the first step, all the
            # prefills turn into decodes and max_query_len will be 1.
            if self.is_multi_step_chunked_prefill and max_query_len == 1:
                query_lens = [1] * len(seq_lens)
            else:
                query_lens = attn_metadata.query_lens

            seq_lens_np = np.array(seq_lens, dtype=np.int32)
            query_lens_np = np.array(query_lens, dtype=np.int32)
            kv_cache_lens = seq_lens_np - query_lens_np
            is_prefill = kv_cache_lens.max() == 0
            is_ringmla_chunked = self.use_ringmla and \
                                 attn_metadata.num_decode_tokens == 0 and \
                                 bool(kv_cache_lens.max() > 0)
            context_lens_tensor = ms.from_numpy(kv_cache_lens)
        else:
            # V1
            is_prefill = attn_metadata.max_context_lens == 0
            is_ringmla_chunked = \
                self.use_ringmla and not is_prefill and \
                bool((attn_metadata.context_lens - \
                      attn_metadata.num_prompt_tokens).min() < 0)
            query_lens_np = attn_metadata.q_seq_lens_np
            seq_lens_np = attn_metadata.seq_lens_np
            context_lens_tensor = attn_metadata.context_lens

        q_seq_lens = ms.Tensor(query_lens_np, dtype=ms.int32)
        position_ids = ms.Tensor(positions, dtype=ms.int32)
        attention_mask = self.casual_mask.gen_attention_mask(
            is_prefill, position_ids, query_lens_np, seq_lens_np)

        model_inputs = {}
        model_inputs["input_ids"] = input_ids.astype(ms.int32)
        model_inputs["batch_valid_length"] = ms.from_numpy(seq_lens_np)
        model_inputs["block_tables"] = attn_metadata.block_tables
        model_inputs["slot_mapping"] = attn_metadata.slot_mapping
        model_inputs["positions"] = position_ids
        model_inputs["q_seq_lens"] = q_seq_lens
        model_inputs["attention_mask"] = attention_mask
        model_inputs["key_cache"] = key_cache
        model_inputs["value_cache"] = value_cache
        model_inputs["context_lens_tensor"] = context_lens_tensor
        model_inputs = (self.update_padding_index_to_inputs(
            model_inputs, q_seq_lens))

        return model_inputs, is_prefill, is_ringmla_chunked

    def forward(self,
                input_ids: Tensor,
                positions: Tensor,
                intermediate_tensors: Optional[IntermediateTensors] = None,
                inputs_embeds: Optional[Tensor] = None,
                **kwargs) -> Union[Tensor, IntermediateTensors]:
        """
        Forward pass of model with support for different computation phases.
        Handles both prefill (context encoding) and incremental
        (token generation) phases.
        Optional RingMLA chunked computation phases for use-MLA model with
        quantization and tensor parallel size < 16.
        Notes:
            - Automatically detects prefill vs incremental phases based on
              input characteristics.
            - Supports RingMLA chunked computation for for use-MLA model.
            - Maintains phase-specific flags for proper graph compilation
              and execution.
        """
        model_inputs, is_prefill, is_ringmla_chunked = self.prepare_inputs(
            input_ids, positions)
        model_inputs = self.update_model_inputs(model_inputs, **kwargs)
        if intermediate_tensors is not None:
            model_inputs["hidden_states"] = \
                intermediate_tensors["hidden_states"]

        if is_prefill or is_ringmla_chunked:
            self.network.phase = \
                "prefill" if not is_ringmla_chunked else "chunked"
            if not self.set_flags or is_pynative():
                self.network.add_flags_custom_mcore(is_prefill=True)
                if hasattr(self.network, 'add_flags_chunked'):
                    # chunked means 3-rd graph "chunked"
                    self.network.add_flags_chunked(
                        is_chunked=is_ringmla_chunked)
                # ringmla_chunked means computing chunked-prefills on ringmla
                self.is_chunked |= is_ringmla_chunked
            hidden_states = self.network(**model_inputs)
            self.network.phase = "increment"
            if not self.set_flags or is_pynative():
                self.network.add_flags_custom_mcore(is_prefill=False)
                self.set_flags = (True
                                  if not self.use_ringmla else self.is_chunked)
        else:
            hidden_states = self.network(**model_inputs)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states})
        return hidden_states

    def _create_network(self):
        # Initial network
        if self.model_config.enforce_eager:
            os.environ['ENFORCE_EAGER'] = 'True'
        with no_init_parameters():  # Delay initialization
            network: PreTrainedModel = AutoModel.from_config(self.mf_config)
            network.model.return_hidden_states = True
        if get_pp_group().is_last_rank:
            return network, network.model.output_layer
        return network, None

    def update_model_inputs(self, model_inputs, **kwargs):
        return model_inputs

    def compute_logits(
        self,
        hidden_states: Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[Tensor]:
        if sampling_metadata is not None:
            selected_token_indices = sampling_metadata.selected_token_indices
            if (selected_token_indices is not None
                    and selected_token_indices.numel() <= 0):
                logits = ms.mint.zeros(
                    (0, self.model_config.hf_config.vocab_size),
                    dtype=self.model_config.hf_config.torch_dtype)
                return logits
            else:
                hidden_states = hidden_states.reshape(
                    (-1, hidden_states.shape[-1]))
                hidden_states = hidden_states.index_select(
                    0, selected_token_indices)
        if is_310p():
            # To get better performance in 310p, the lm head should run
            # in O0 mode to avoid transdata, 910 keep the original process.
            if self.lm_head_graph is None:
                self.lm_head_graph = ms.jit(function=self.lm_head,
                                            jit_level="O0")
            logits = self.lm_head_graph(hidden_states)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.view(-1, logits.shape[-1])
        return logits

    def load_weights(self, weights: Iterable[tuple[str, Tensor]]):
        self.network.load_weights(self.mf_config.load_checkpoint)
        return None
