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

import numpy as np

from vllm.attention import AttentionMetadata
from vllm.config import VllmConfig
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.logger import init_logger


from mindformers.tools.register.config import MindFormerConfig

from mindformers.core.context import build_context
from mindformers.core.parallel_config import build_parallel_config
from mindformers.trainer.utils import transform_and_load_checkpoint
from research.deepseek3.deepseek3_config import (
    DeepseekV3Config as DeepseekV3Config_MF,
)
from research.deepseek3.deepseek3 import (
    DeepseekV3ForCausalLM as DeepseekV3ForCausalLM_MF,
)

from vllm_mindspore.model_executor.layers.sampler import get_sampler
from vllm_mindspore.model_executor.models.model_base import MsModelBase
from vllm_mindspore.utils import cal_block_num

import mindspore as ms
from mindspore import Tensor, JitConfig, Model


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

        self.mf_model_config = DeepseekV3Config_MF(**self.mf_config.model.model_config)
        self.mf_model_config.num_blocks = cal_block_num(self.cache_config, self.model_config, self.parallel_config)
        self.mf_model_config.block_size = self.cache_config.block_size
        if self.mf_config.moe_config:
            self.mf_model_config.moe_config = self.mf_config.moe_config

        # Initital network
        self.network = DeepseekV3ForCausalLM_MF(self.mf_model_config)

        # quant
        if hasattr(self.mf_model_config, "quantization_config") and self.mf_model_config.quantization_config:        
            from mindspore_gs.ptq import PTQ
            from mindspore_gs.ptq import PTQMode, PTQConfig, OutliersSuppressionType, PrecisionRecovery, QuantGranularity
            from mindspore_gs.common import BackendTarget
            from mindspore.common import dtype as msdtype
            from collections import OrderedDict
            cfg = PTQConfig(mode=PTQMode.DEPLOY,
                            backend=BackendTarget.ASCEND,
                            weight_quant_dtype=msdtype.int8,
                            act_quant_dtype=msdtype.int8,
                            outliers_suppression=OutliersSuppressionType.OUTLIER_SUPPRESSION_PLUS,
                            opname_blacklist=['lkv2kv', 'lm_head'],
                            precision_recovery=PrecisionRecovery.NONE,
                            act_quant_granularity=QuantGranularity.PER_TENSOR,
                            weight_quant_granularity=QuantGranularity.PER_CHANNEL)
            wo_config = PTQConfig(mode=PTQMode.DEPLOY,
                                  backend=BackendTarget.ASCEND,
                                  weight_quant_dtype=msdtype.int8,
                                  act_quant_dtype=msdtype.int8,
                                  outliers_suppression=OutliersSuppressionType.NONE,
                                  precision_recovery=PrecisionRecovery.NONE,
                                  act_quant_granularity=QuantGranularity.PER_TENSOR,
                                  weight_quant_granularity=QuantGranularity.PER_CHANNEL)
            ffn_config = PTQConfig(mode=PTQMode.DEPLOY,
                                   backend=BackendTarget.ASCEND,
                                   weight_quant_dtype=msdtype.int8,
                                   act_quant_dtype=msdtype.int8,
                                   outliers_suppression=OutliersSuppressionType.NONE,
                                   precision_recovery=PrecisionRecovery.NONE,
                                   act_quant_granularity=QuantGranularity.PER_TOKEN,
                                   weight_quant_granularity=QuantGranularity.PER_CHANNEL)
            ptq = PTQ(config=cfg,
                      layer_policies=OrderedDict({r'.*\.wo.*':wo_config, r'.*\.feed_forward\..*':ffn_config}))
            ptq.apply(self.network)
            ptq.convert(self.network)

        self.network._jit_config_dict = JitConfig(
            jit_level="O0", infer_boost="on"
        ).jit_config_dict
        self.mf_kvcaches_init = False
        self.logits = None

        self.sampler = get_sampler()
        self.set_modules({"model": self.network})

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

        is_prefill = True if attn_metadata.prefill_metadata else False

        self.logits = None

        model_inputs = {}
        model_inputs["input_ids"] = _batch_seq(input_ids, is_prefill)
        model_inputs["batch_valid_length"] = ms.ops.expand_dims(
            attn_metadata.seq_lens_tensor, 0
        )
        model_inputs["block_tables"] = _pad_block_table(
            attn_metadata.block_tables,
            self.mf_model_config.seq_length,
            self.mf_model_config.block_size,
        )
        model_inputs["slot_mapping"] = attn_metadata.slot_mapping

        if is_prefill:
            self.network.phase = "prefill"
            self.network.add_flags_custom(is_first_iteration=True)
            self.logits = self.network(**model_inputs)
            self.network.phase = "increment"
            self.network.add_flags_custom(is_first_iteration=False)
        else:
            self.logits = self.network(**model_inputs)

        return None

    def compute_logits(
        self,
        hidden_states: Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[Tensor]:
        return self.logits

    def sample(
        self,
        logits: Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str, Tensor]]) -> Set[str]:
        model = Model(self.network)
        batch_size = self.mf_config.model.model_config.batch_size
        seq_length = self.mf_config.model.model_config.seq_length
        input_ids = np.ones(shape=tuple([batch_size, seq_length]))
        infer_data = self.network.prepare_inputs_for_predict_layout(input_ids)
        transform_and_load_checkpoint(
            self.mf_config, model, self.network, infer_data, do_predict=True
        )
        self.network.set_dynamic_inputs()
        return None
