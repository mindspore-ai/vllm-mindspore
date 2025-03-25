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

import os
from typing import Iterable, List, Optional, Set, Tuple, Union
from pathlib import Path
from collections import OrderedDict

import numpy as np

from vllm.attention import AttentionMetadata
from vllm.config import VllmConfig
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.logger import init_logger


from mindformers.tools.register.config import MindFormerConfig
from mindspore.common.api import _pynative_executor
from mindformers.core.context import build_context
from mindformers.core.parallel_config import build_parallel_config
from mindformers.trainer.utils import transform_and_load_checkpoint
from research.deepseek3.deepseek3_model_infer import DeepseekV3DecodeLayer
from research.deepseek3.deepseek3_config import (
    DeepseekV3Config as DeepseekV3Config_MF,
)
from research.deepseek3.deepseek3 import (
    DeepseekV3ForCausalLM as DeepseekV3ForCausalLM_MF,
)

from vllm_mindspore.model_executor.layers.sampler import get_sampler
from vllm_mindspore.model_executor.models.model_base import MsModelBase
from vllm_mindspore.utils import calc_block_num

import mindspore as ms
from mindspore import Tensor, JitConfig, Model
from mindspore.common import dtype as msdtype

from mindspore_gs.ptq import PTQ
from mindspore_gs.ptq import PTQMode, PTQConfig, OutliersSuppressionType, PrecisionRecovery, QuantGranularity, GPTQQuantConfig
from mindspore_gs.common import BackendTarget

from vllm_mindspore.model_executor.models.mf_models.deepseekv3_infer_parallelism import DeepseekInferParallelism


logger = init_logger(__name__)

def _pad_to_max(x, max_len):
    return x + [-1] * (max_len - len(x))


def _pad_block_table(block_tables, seq_length, block_size):
    # When prefill, the block_tables is a empty tensor.
    if len(block_tables.shape) < 2:
        fake_block_tables = ms.mint.empty(
            2, seq_length // block_size, dtype=ms.int32, device="Ascend"
        )
        return fake_block_tables

    block_tables_list = block_tables.tolist()
    padded_block_tables = [
        _pad_to_max(block_table, seq_length // block_size)
        for block_table in block_tables_list
    ]

    return Tensor(np.array(padded_block_tables).astype(np.int32))


def _batch_seq(input_tokens, prefill):
    if prefill:
        return ms.ops.expand_dims(input_tokens, 0).to(ms.int32)

    return ms.mint.reshape(input_tokens, (-1, 1)).to(ms.int32)


class DeepseekV3ForCausalLM(MsModelBase):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super(DeepseekV3ForCausalLM, self).__init__(
            vllm_config=vllm_config, prefix=prefix
        )

        self.mf_config = MindFormerConfig(os.getenv("MINDFORMERS_MODEL_CONFIG"))
        build_context(self.mf_config, is_set_ms_ctx=False, is_init_ms=False)
        build_parallel_config(self.mf_config)
        self.mf_config.model.model_config.parallel_config = (
            self.mf_config.parallel_config
        )
        self.mf_config.model.model_config.parallel_config.model_parallel = (
            get_tensor_model_parallel_world_size()
        )
        self.mf_config.model.model_config.parallel_config.pipeline_stage = 1
        self.mf_config.load_checkpoint = self.get_model_path()

        self.mf_model_config = DeepseekV3Config_MF(**self.mf_config.model.model_config)
        self.mf_model_config.num_blocks = calc_block_num(self.cache_config, self.model_config, self.parallel_config)
        self.mf_model_config.block_size = self.cache_config.block_size
        if self.mf_config.moe_config:
            self.mf_model_config.moe_config = self.mf_config.moe_config
        self.mf_model_config.return_hidden_states = True

        self.is_quant = bool(hasattr(self.mf_model_config, "quantization_config") and
                             self.mf_model_config.quantization_config)
        # Initital network
        self.network = DeepseekV3ForCausalLM_MF(self.mf_model_config)

        # quant
        if hasattr(self.mf_model_config, "quantization_config") and hasattr(self.mf_model_config.quantization_config, "quant_method"):
            ptq = self.create_ptq(self.mf_model_config.quantization_config.quant_method, PTQMode.DEPLOY)
            if ptq is not None:
                ptq.apply(self.network)
                ptq.convert(self.network)

        self.network._jit_config_dict = JitConfig(
            jit_level="O0", infer_boost="on"
        ).jit_config_dict
        self.mf_kvcaches_init = False

        self.sampler = get_sampler()
        self.set_modules({"model": self.network})

        self.prefill_mask = Tensor(np.triu(np.ones(shape=(128, 128), dtype=np.float16), k=1), dtype=ms.bfloat16)

        self.decode_mask = Tensor(np.triu(np.ones(shape=(self.mf_model_config.seq_length,
                                                         self.mf_model_config.seq_length), dtype=np.int8), k=1),
                                  dtype=ms.bfloat16)

        self.hard_mask = Tensor([0], dtype=ms.bfloat16).reshape(1, 1)

        self.gather = ms.ops.Gather()

        self.set_flags = False

    def update_mf_kvcaches(self, kv_caches):
        if self.mf_kvcaches_init:
            return

        for i in range(self.mf_model_config.num_layers):
            k_cache = kv_caches[i][0]
            mf_k_cache, _ = self.network.kvcache(i)

            mf_k_cache.set_device_address(
                k_cache._data_ptr(), k_cache.shape, k_cache.dtype
            )
        self.mf_kvcaches_init = True

    def gen_attention_mask(self, is_prefill, position_ids, query_lens):
        if is_prefill:
            attention_mask = self.prefill_mask
        else:
            if max(query_lens) > 1:
                attention_mask = self.gather(self.decode_mask, position_ids, 0)
            else:
                attention_mask = self.hard_mask
        return attention_mask

    def forward(
        self,
        input_ids: Tensor,
        positions: Tensor,
        kv_caches: List[Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[Tensor] = None,
    ) -> Union[Tensor, IntermediateTensors]:
        self.update_mf_kvcaches(kv_caches)
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
        if attn_metadata.num_decode_tokens == 0 and kv_cache_lens.max() == 0:
            is_prefill = True
        else:
            is_prefill = False

        q_seq_lens = ms.Tensor(query_lens, dtype=ms.int32)
        position_ids = ms.Tensor(positions, dtype=ms.int32)
        attention_mask = self.gen_attention_mask(is_prefill, position_ids, query_lens)

        model_inputs = {}
        model_inputs["input_ids"] = _batch_seq(input_ids, is_prefill)
        model_inputs["batch_valid_length"] = ms.Tensor.from_numpy(np.expand_dims(seq_lens_np, 0))
        model_inputs["block_tables"] = _pad_block_table(
            attn_metadata.block_tables,
            self.mf_model_config.seq_length,
            self.mf_model_config.block_size,
        )
        model_inputs["slot_mapping"] = attn_metadata.slot_mapping
        model_inputs["position_ids"] = position_ids
        model_inputs["q_seq_lens"] = q_seq_lens
        model_inputs["attention_mask"] = attention_mask
        _pynative_executor.sync()
        if is_prefill:
            self.network.phase = "prefill"
            if not self.set_flags:
                self.network.add_flags_custom(is_first_iteration=True)
            hidden_states = self.network(**model_inputs)
            self.network.phase = "increment"
            if not self.set_flags:
                self.network.add_flags_custom(is_first_iteration=False)
                self.set_flags = True
        else:
            hidden_states = self.network(**model_inputs)

        return hidden_states

    def compute_logits(
        self,
        hidden_states: Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[Tensor]:
        selected_token_indices = sampling_metadata.selected_token_indices
        if selected_token_indices is not None and selected_token_indices.numel() <= 0:
            logits = ms.mint.zeros((0, self.mf_model_config.vocab_size),
                                    dtype=self.mf_model_config.compute_dtype)
        else:
            hidden_states = hidden_states.index_select(0, selected_token_indices)
            logits = self.network.lm_head(hidden_states)
            logits = logits.reshape(-1, logits.shape[-1])

        return logits

    def sample(
        self,
        logits: Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str, Tensor]]) -> Set[str]:
        if self.mf_config.load_ckpt_format == "ckpt":
            model = Model(self.network)
            batch_size = self.mf_config.model.model_config.batch_size
            seq_length = self.mf_config.model.model_config.seq_length
            input_ids = np.ones(shape=tuple([batch_size, seq_length]))
            infer_data = self.network.prepare_inputs_for_predict_layout(input_ids)
            transform_and_load_checkpoint(
                self.mf_config, model, self.network, infer_data, do_predict=True
            )
        else:
            model_parallelism = DeepseekInferParallelism(self.mf_config, self.network, self.is_quant)
            model_parallelism.infer_convert_and_parallelism(self.mf_config.load_checkpoint)

        self.network.set_dynamic_inputs()
        dynamic_hidden_states = ms.Tensor(shape=[None, None], dtype=self.mf_model_config.compute_dtype)
        self.network.lm_head.set_inputs(dynamic_hidden_states)
        return None

    def get_model_path(self):
        model_name_or_path = self.model_config.model
        if os.path.isdir(model_name_or_path):
            return model_name_or_path
        else:
            raise ValueError("The 'model' in LLM should be the local path of the MindSpore checkpoint file.")

    def create_ptq(self, quant_type: str, quant_mode: PTQMode):
        """create_ptq"""
        if quant_type.lower() == 'ptq':
            cfg = PTQConfig(mode=quant_mode, backend=BackendTarget.ASCEND, weight_quant_dtype=msdtype.int8,
                            act_quant_dtype=msdtype.int8,
                            outliers_suppression=OutliersSuppressionType.OUTLIER_SUPPRESSION_PLUS,
                            opname_blacklist=['lkv2kv', 'lm_head'], precision_recovery=PrecisionRecovery.NONE,
                            act_quant_granularity=QuantGranularity.PER_TENSOR,
                            weight_quant_granularity=QuantGranularity.PER_CHANNEL)
            ffn_config = PTQConfig(mode=quant_mode, backend=BackendTarget.ASCEND, weight_quant_dtype=msdtype.int8,
                                act_quant_dtype=msdtype.int8,
                                outliers_suppression=OutliersSuppressionType.NONE,
                                precision_recovery=PrecisionRecovery.NONE,
                                act_quant_granularity=QuantGranularity.PER_TOKEN,
                                weight_quant_granularity=QuantGranularity.PER_CHANNEL)
            layer_policies = OrderedDict({r'.*\.feed_forward\..*': ffn_config})
        elif quant_type.lower() == 'awq-a16w4':
            cfg = PTQConfig(mode=quant_mode, backend=BackendTarget.ASCEND, weight_quant_dtype=msdtype.qint4x2,
                            act_quant_dtype=None, outliers_suppression=OutliersSuppressionType.AWQ,
                            opname_blacklist=['lm_head', 'lkv2kv'], weight_quant_granularity=QuantGranularity.PER_GROUP,
                            group_size=128)
            layer_policies = OrderedDict()
        elif quant_type.lower() == 'awq-a16w8':
            cfg = PTQConfig(mode=quant_mode, backend=BackendTarget.ASCEND, weight_quant_dtype=msdtype.int8,
                            act_quant_dtype=None, outliers_suppression=OutliersSuppressionType.AWQ,
                            opname_blacklist=['lm_head', 'lkv2kv'])
        elif quant_type.lower() == 'gptq-perchannel':
            gptq_config = GPTQQuantConfig()
            cfg = PTQConfig(mode=quant_mode, backend=BackendTarget.ASCEND, weight_quant_dtype=msdtype.qint4x2,
                            act_quant_dtype=None, precision_recovery=PrecisionRecovery.GPTQ, algo_args=gptq_config,
                            opname_blacklist=['lm_head', 'lkv2kv'])
            layer_policies = OrderedDict()
        elif quant_type.lower() == 'gptq-pergroup':
            gptq_config = GPTQQuantConfig()
            cfg = PTQConfig(mode=quant_mode, backend=BackendTarget.ASCEND, weight_quant_dtype=msdtype.qint4x2,
                            algo_args=gptq_config, act_quant_dtype=None, precision_recovery=PrecisionRecovery.GPTQ,
                            weight_quant_granularity=QuantGranularity.PER_GROUP, opname_blacklist=['lm_head', 'lkv2kv'],
                            group_size=128)
            layer_policies = OrderedDict()
        elif quant_type.lower() == 'smoothquant':
            cfg = PTQConfig(mode=quant_mode, backend=BackendTarget.ASCEND, weight_quant_dtype=msdtype.int8,
                            act_quant_dtype=msdtype.int8, outliers_suppression=OutliersSuppressionType.SMOOTH,
                            opname_blacklist=['lm_head', 'lkv2kv'])
            w2_config = PTQConfig(mode=quant_mode, backend=BackendTarget.ASCEND, weight_quant_dtype=msdtype.int8,
                                act_quant_dtype=msdtype.int8,
                                outliers_suppression=OutliersSuppressionType.NONE,
                                precision_recovery=PrecisionRecovery.NONE,
                                act_quant_granularity=QuantGranularity.PER_TOKEN,
                                weight_quant_granularity=QuantGranularity.PER_CHANNEL)
            layer_policies = OrderedDict({r'.*\.w2.*': w2_config})
        elif quant_type.lower() == 'a16w8':
            cfg = PTQConfig(mode=quant_mode, backend=BackendTarget.ASCEND, weight_quant_dtype=msdtype.int8,
                            opname_blacklist=['lm_head', 'lkv2kv'])
            layer_policies = OrderedDict()
        elif quant_type.lower() == 'a8dynw8':
            cfg = PTQConfig(mode=quant_mode, backend=BackendTarget.ASCEND, weight_quant_dtype=msdtype.int8,
                            act_quant_dtype=msdtype.int8, act_quant_granularity=QuantGranularity.PER_TOKEN,
                            opname_blacklist=['lm_head', 'lkv2kv'])
            layer_policies = OrderedDict()
        else:
            logger.warning("Input unsupported quant type: %s.", quant_type)
            return None
        ptq = PTQ(config=cfg, layer_policies=layer_policies)
        if 'awq' in quant_type.lower():
            # pylint: disable=protected-access
            ptq._config.weight_symmetric = False
        ptq.decoder_layer_types.append(DeepseekV3DecodeLayer)
        return ptq