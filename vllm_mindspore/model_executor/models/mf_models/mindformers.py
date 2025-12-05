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
from mindformers.core.context import build_mf_context, build_parallel_context
from mindspore import Tensor, mutable, ops
from mindspore.common.api import _no_grad as no_grad
from mindspore.nn.utils import no_init_parameters
from vllm.config import CompilationLevel, VllmConfig, get_current_vllm_config
from vllm.distributed.parallel_state import get_dp_group, get_pp_group
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import SupportsPP
from vllm.sequence import IntermediateTensors

from vllm_mindspore.model_executor.models.attention_mask import (
    LowerTriangularMask, MLALowerTriangularMask)
from vllm_mindspore.model_executor.models.mf_models.config import gen_mf_config
from vllm_mindspore.model_executor.models.model_base import (
    AttentionWrapper, MLAAttentionWrapper, MsModelBase)
from vllm_mindspore.model_executor.models.utils import (
    convert_pin, is_use_ringmla, make_empty_intermediate_tensors_factory)
from vllm_mindspore.utils import is_310p

logger = init_logger(__name__)


class MindFormersForCausalLM(MsModelBase, SupportsPP):
    _set_launch_group = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.model_config = vllm_config.model_config
        self.lm_head_graph = None
        self.is_eager_mode = vllm_config.model_config.enforce_eager

        self.enable_aclgraph = \
            vllm_config.compilation_config.level == CompilationLevel.PIECEWISE

        mf_config = gen_mf_config(vllm_config)
        mf_config.load_checkpoint = self.get_model_path()
        mf_config.pretrained_model_dir = self.get_model_path()
        self.mf_config = mf_config
        self.mla_config = self.mf_config.get('model', None).get(
            'model_config', None).get('multi_latent_attention', False)
        self.use_ringmla = is_use_ringmla(vllm_config, mf_config)
        self.mf_config.model.model_config.use_fused_mla = self.use_ringmla

        build_mf_context(self.mf_config)
        mf_par_ctx = build_parallel_context(self.mf_config)
        mf_par_ctx.init_communication()
        if self.mla_config and not self.enable_aclgraph:
            self._set_runtime_kernel_launch_group()

        self.network, self.lm_head = self._create_network()
        self.casual_mask = self._create_mask()
        if hasattr(self.network, "quant_config") and self.network.quant_config:
            self.fa3_quant = self.network.quant_config.fa3_quant
            self.fa3_quant_layer = self.network.quant_config.fa3_quant_layer
            # used when allocate the kvcache in GPUModelRunner
            vllm_config.quant_config = self.network.quant_config
        else:
            self.fa3_quant = False
            self.fa3_quant_layer = set()
        if self.fa3_quant and vllm_config.cache_config.cache_dtype == "auto":
            raise RuntimeError(
                "To use fa3_quant, it is necessary to set "
                "\"--kv-cache-dtype 'int8'\" in the startup command.")
        if self.fa3_quant and not self.use_ringmla:
            raise ValueError('To use fa3_quant, '
                             'it is necessary to set use_ringmla to True.')
        if not self.fa3_quant and \
                    vllm_config.cache_config.cache_dtype == "int8":
            raise RuntimeError("To use kv-cache-dtype 'int8', "
                               "it is necessary to set fa3_quant to True.")
        if self.fa3_quant:
            logger.info("fa_quant is enabled, quant_layer is %s.",
                        self.fa3_quant_layer)
        self._set_dynamic_inputs()

        self.set_modules({"model": self.network})

        self.kv_caches = self._create_kv_caches()
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")

        for i in range(self.num_layers):
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

    def _create_kv_caches(self):
        # Initial kv_caches
        wrapper_func = (MLAAttentionWrapper
                        if self.mla_config else AttentionWrapper)
        if self.fa3_quant:
            return [wrapper_func(fa3_quant=True, kv_cache_dtype=ms.int8) \
                    if self.fa3_quant and i in self.fa3_quant_layer \
                    else wrapper_func(fa3_quant=True,
                                      kv_cache_dtype=self.model_config.dtype) \
                        for i in range(self.num_layers)]
        else:
            return [wrapper_func() for _ in range(self.num_layers)]

    def get_kvcache(self):
        if not self.mla_config:
            return super().get_kvcache()

        forward_context = get_forward_context()
        key_cache = [
            self.kv_caches[i].kv_cache[forward_context.virtual_engine][0]
            for i in range(self.num_layers)
        ]
        if not self.use_ringmla:
            return mutable(key_cache), None
        # deepseek mla op need key cache and rope cache
        rope_cache = [
            self.kv_caches[i].kv_cache[forward_context.virtual_engine][1]
            for i in range(self.num_layers)
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

        model_inputs["attn_padding_idx"] = convert_pin(attn_padding_idx)
        model_inputs["attn_unpadding_idx"] = convert_pin(attn_unpadding_idx)
        model_inputs["ffn_padding_idx"] = convert_pin(ffn_padding_idx)
        model_inputs["ffn_unpadding_idx"] = convert_pin(ffn_unpadding_idx)

        return model_inputs

    def prepare_inputs(self, input_ids, positions):

        attn_metadata = get_forward_context().attn_metadata
        # 0.9.1 attn_metadata[layer_name], don't have layer_name here
        # so we just take one by default
        if isinstance(attn_metadata, dict) and '1' in attn_metadata:
            attn_metadata = attn_metadata['1']

        if attn_metadata is None:
            # To enable dummy_run in chunked scenarios, set q_seq_lens_np
            # to be greater than 1.
            if self.has_prefill_warmup and not self.has_chunked_warmup:
                input_ids = ms.ops.cat((input_ids, input_ids))
                positions = ms.ops.cat((positions, ms.tensor([1])))

            attn_metadata = self._dummy_attention_metadata(
                input_ids, positions)

        key_cache, value_cache = self.get_kvcache()

        is_prefill = attn_metadata.max_context_lens == 0
        is_ringmla_chunked = \
            self.use_ringmla and not is_prefill and \
            bool(attn_metadata.q_seq_lens_np.max() > 1)
        query_lens_np = attn_metadata.q_seq_lens_np
        seq_lens_np = attn_metadata.seq_lens_np
        context_lens_tensor = attn_metadata.context_lens

        q_seq_lens = ms.Tensor(query_lens_np, dtype=ms.int32)
        position_ids = ms.Tensor(positions, dtype=ms.int32)
        attention_mask = self.casual_mask.gen_attention_mask(
            is_prefill, position_ids, query_lens_np, seq_lens_np)

        model_inputs = {}
        model_inputs["input_ids"] = convert_pin(input_ids.astype(ms.int32))
        model_inputs["batch_valid_length"] = convert_pin(
            ms.from_numpy(seq_lens_np))
        model_inputs["block_tables"] = convert_pin(attn_metadata.block_tables)
        model_inputs["slot_mapping"] = convert_pin(attn_metadata.slot_mapping)
        model_inputs["positions"] = convert_pin(position_ids)
        model_inputs["q_seq_lens"] = convert_pin(q_seq_lens)
        model_inputs["attention_mask"] = convert_pin(attention_mask)
        model_inputs["key_cache"] = key_cache
        model_inputs["value_cache"] = value_cache
        model_inputs["context_lens_tensor"] = convert_pin(context_lens_tensor)
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
                convert_pin(intermediate_tensors["hidden_states"])
        elif kwargs.get("previous_hidden_states") is not None:
            # used for deepseek-mtp
            model_inputs["hidden_states"] = convert_pin(
                kwargs["previous_hidden_states"])

        def _set_network_flags(prefill_flag, chunked_flag):
            self.network.add_flags_custom_mcore(is_prefill=prefill_flag)
            if hasattr(self.network, "add_flags_chunked"):
                self.network.add_flags_chunked(
                    is_chunked=(chunked_flag and is_ringmla_chunked))

        if self.is_eager_mode:
            # In eager_mode, there is no need to set flags repeatedly in
            # decoding, until there is new prefill or chunked prediction.
            need_set_flag = is_prefill or is_ringmla_chunked
        else:
            # In graph_mode, there is no need to set flags until all inference
            # stages have been executed (including prefill/decode,
            # and chunked only if ringmla is enabled).
            need_set_flag = (not self.has_prefill_warmup
                             or not self.has_chunked_warmup)
            self.network.phase =  "prefill" if is_prefill \
                else "chunked" if is_ringmla_chunked else "increment"

        # The value of has_prefill_warmup and has_chunked_warmup indicates
        # whether the corresponding inference graph has been executed.
        # If ringmla is disabled, the value of has_chunked_warmup would be
        # initialized to True, indicating that there is no need to execute
        # chunked graph.
        if need_set_flag:
            _set_network_flags(True, True)
            hidden_states = self.network(**model_inputs)
            _set_network_flags(False, False)
            self.has_prefill_warmup = True
            self.has_chunked_warmup = (not self.use_ringmla
                                       or is_ringmla_chunked)
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

    @classmethod
    def _set_runtime_kernel_launch_group(cls):
        """Set the parameters of kernel_launch_group"""
        logger.info("........ Enable kernel_launch_group ........")
        if cls._set_launch_group:
            return
        thread_num = 4
        kernel_group_num = 16

        ms.runtime.set_kernel_launch_group(thread_num=thread_num,
                                           kernel_group_num=kernel_group_num)
        cls._set_launch_group = True

    def compute_logits(
        self,
        hidden_states: Tensor,
    ) -> Optional[Tensor]:
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

    def capture_start_time(self, weights: Iterable[tuple[str, Tensor]]):
        """A hook to capture the start time of loading weights."""
        # To capture the start time of loading weights,
        # we break after the first iteration.
        next(weights, None)

    @no_grad()
    def load_weights(self, weights: Iterable[tuple[str, Tensor]]):
        self.capture_start_time(weights)
        self.network.load_weights(self.mf_config.load_checkpoint)
        return None
