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
from typing import Iterable, Set, Tuple
from collections import OrderedDict

import numpy as np

from vllm.config import VllmConfig
from vllm.config import get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger

from mindspore import Tensor, JitConfig, Model, mutable
from mindspore.common import dtype as msdtype
from mindspore.nn.utils import no_init_parameters

from mindspore_gs.ptq import PTQ
from mindspore_gs.ptq import PTQMode, PTQConfig, OutliersSuppressionType, PrecisionRecovery, QuantGranularity, \
    GPTQQuantConfig
from mindspore_gs.common import BackendTarget

from mindformers.trainer.utils import transform_and_load_checkpoint
from research.deepseek3.deepseek3_model_infer import DeepseekV3DecodeLayer
from research.deepseek3.deepseek3_config import (
    DeepseekV3Config as DeepseekV3Config_MF,
)
from research.deepseek3.deepseek3 import (
    DeepseekV3ForCausalLM as DeepseekV3ForCausalLM_MF,
)

from vllm_mindspore.model_executor.layers.sampler import get_sampler
from vllm_mindspore.model_executor.models.mf_models.mf_model_base import MfModelBase, Fake_MLA

from vllm_mindspore.model_executor.models.mf_models.deepseekv3_weight_processor import DeepseekV3WeightProcessor
from vllm_mindspore.model_executor.models.mf_models.attention_mask import LowerTriangularMask

logger = init_logger(__name__)


class DeepseekV3ForCausalLM(MfModelBase):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super(DeepseekV3ForCausalLM, self).__init__(
            vllm_config=vllm_config, prefix=prefix
        )

        self.mf_config.load_checkpoint = self.get_model_path()

        self.mf_model_config = DeepseekV3Config_MF(**self.mf_config.model.model_config)
        if self.mf_config.moe_config:
            self.mf_model_config.moe_config = self.mf_config.moe_config
        self.mf_model_config.return_hidden_states = True
        setattr(self.mf_model_config, 'npu_mem_size', -1)

        self.is_quant = bool(hasattr(self.mf_model_config, "quantization_config") and
                             self.mf_model_config.quantization_config)
        # Initital network
        with no_init_parameters():  # Delay initialization
            self.network = DeepseekV3ForCausalLM_MF(self.mf_model_config)

        # quant
        if hasattr(self.mf_model_config, "quantization_config") and hasattr(self.mf_model_config.quantization_config,
                                                                            "quant_method"):
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

        self.kv_caches = [Fake_MLA() for i in range(self.mf_model_config.num_layers)]
        compilation_config = get_current_vllm_config().compilation_config

        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        for i in range(self.mf_model_config.num_layers):
            compilation_config.static_forward_context[str(i)] = self.kv_caches[i]

        self.casual_mask = LowerTriangularMask(mf_model_config=self.mf_model_config)
        self.set_flags = False

    def get_kvcache(self):
        key_cache = []
        forward_context = get_forward_context()
        for i in range(self.mf_model_config.num_layers):
            k_cache = self.kv_caches[i].kv_cache[forward_context.virtual_engine][0]
            key_cache.append(k_cache)
        return mutable(key_cache), None

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
            weight_processor = DeepseekV3WeightProcessor(self.mf_config, self.network, self.is_quant)
            weight_processor.load_safetensors_shard(self.mf_config.load_checkpoint)
        self.network.set_dynamic_inputs()
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
        if 'smoothquant' in quant_type.lower():
            # pylint: disable=protected-access
            ptq._config.aclnn_quant_list = ["routed_experts.ffn.w_gate_hidden"]
        ptq.decoder_layer_types.append(DeepseekV3DecodeLayer)
        return ptq
