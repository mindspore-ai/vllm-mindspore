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
import vllm.envs as envs
import mindspore as ms

from vllm.config import VllmConfig
from vllm.config import get_current_vllm_config
from vllm.distributed.parallel_state import get_dp_group, get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger

import mindspore as ms
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
from vllm_mindspore.model_executor.models.model_base import Fake_MLA, Fake_MLA_V1
from vllm_mindspore.model_executor.models.mf_models.mf_model_base import MfModelBase
from vllm_mindspore.model_executor.models.mf_models.deepseekv3_weight_processor import DeepseekV3WeightProcessor
from vllm_mindspore.model_executor.models.attention_mask import MLALowerTriangularMask

logger = init_logger(__name__)


def set_runtime_kernel_launch_group():
    kernel_launch_group = {'thread_num': 2, 'kernel_group_num': 8}
    env_kernel_launch_group = os.getenv("EXPERIMENTAL_KERNEL_LAUNCH_GROUP", None)
    if env_kernel_launch_group is not None:
        pairs = env_kernel_launch_group.split(',')
        for pair in pairs:
            key, val = pair.split(':')
            kernel_launch_group[key] = val
    thread_num = int(kernel_launch_group.get('thread_num', 2))
    kernel_group_num = int(kernel_launch_group.get('kernel_group_num', 8))
    ms.runtime.set_kernel_launch_group(thread_num=thread_num, kernel_group_num=kernel_group_num)


def _get_padding_index(q_seq_len):
    dp_size = get_dp_group().world_size
    tp_size = get_tensor_model_parallel_world_size()
    if dp_size == 1:
        return None, None, None, None

    tokens_len_per_dp = q_seq_len.sum().reshape(-1)
    tokens_len_per_dp = get_dp_group().all_gather(tokens_len_per_dp)
    tokens_len_per_dp = tokens_len_per_dp.asnumpy()
    padding_size = (tokens_len_per_dp.max() + tp_size - 1) // tp_size * tp_size

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
            pad = np.zeros(padding_size - arange_data.shape[0], dtype=np.int32)
            attn_padding_idx = np.concatenate((arange_data, pad), axis=0)
        if dp_rank == 0:
            attn_unpadding_idx = arange_data
            last_arange_index = arange_data[-1]
            pad = np.zeros(padding_size - attn_unpadding_idx.shape[0], dtype=np.int32)
            ffn_padding_idx = np.concatenate((attn_unpadding_idx, pad), axis=0)
        else:
            attn_offset_idx = arange_data + padding_size * dp_rank
            attn_unpadding_idx = np.concatenate((attn_unpadding_idx, attn_offset_idx), axis=0)
            ffn_offset_idx = arange_data + last_arange_index + 1
            last_arange_index = ffn_offset_idx[-1]
            pad = np.zeros(padding_size - ffn_offset_idx.shape[0], dtype=np.int32)
            ffn_padding_idx = np.concatenate((ffn_padding_idx, ffn_offset_idx, pad), axis=0)
    return ms.from_numpy(attn_padding_idx), ms.from_numpy(attn_unpadding_idx), ms.from_numpy(ffn_padding_idx), \
           ms.from_numpy(ffn_unpadding_idx)



class DeepseekV3ForCausalLM(MfModelBase):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super(DeepseekV3ForCausalLM, self).__init__(
            vllm_config=vllm_config, prefix=prefix
        )
        self.is_quant = bool(hasattr(self.mf_model_config, "quantization_config") and
                             self.mf_model_config.quantization_config)

        self.mf_kvcaches_init = False

        self.sampler = get_sampler()
        self.set_modules({"model": self.network})
        if envs.VLLM_USE_V1:
            self.kv_caches = [Fake_MLA_V1() for i in range(self.mf_model_config.num_layers)]
        else:
            self.kv_caches = [Fake_MLA() for i in range(self.mf_model_config.num_layers)]
        compilation_config = get_current_vllm_config().compilation_config

        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        for i in range(self.mf_model_config.num_layers):
            compilation_config.static_forward_context[str(i)] = self.kv_caches[i]

        self.set_flags = False
        set_runtime_kernel_launch_group()
        self.casual_mask = MLALowerTriangularMask(dtype=self.mf_model_config.compute_dtype,
                                                  max_model_len=self.mf_model_config.seq_length)

    def _generate_model_config(self):
        self.mf_config.load_checkpoint = self.get_model_path()

        self.mf_model_config = DeepseekV3Config_MF(**self.mf_config.model.model_config)
        if self.mf_config.moe_config:
            self.mf_model_config.moe_config = self.mf_config.moe_config
            # dispatch/combine in moe need max_num_seqs as global_max_bs
            if hasattr(self.mf_model_config.moe_config, "dispatch_global_max_bs"):
                self.mf_model_config.moe_config.dispatch_global_max_bs = self.scheduler_config.max_num_seqs
        self.mf_model_config.return_hidden_states = True
        setattr(self.mf_model_config, 'npu_mem_size', -1)

    def _create_network(self):
        # Initital network
        with no_init_parameters():  # Delay initialization
            network = DeepseekV3ForCausalLM_MF(self.mf_model_config)

        # quant
        if hasattr(self.mf_model_config, "quantization_config") and hasattr(self.mf_model_config.quantization_config,
                                                                            "quant_method"):
            ptq = self.create_ptq(self.mf_model_config.quantization_config.quant_method, PTQMode.DEPLOY)
            if ptq is not None:
                ptq.apply(network)
                ptq.convert(network)
        return network, network.lm_head

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
        return None

    def prepare_inputs(self, input_ids, positions, attn_metadata):
        model_inputs, is_prefill = super().prepare_inputs(
            input_ids, positions, attn_metadata)

        attn_padding_idx, attn_unpadding_idx, ffn_padding_idx, ffn_unpadding_idx = _get_padding_index(
            model_inputs["q_seq_lens"])
        model_inputs["attn_padding_idx"] = attn_padding_idx
        model_inputs["attn_unpadding_idx"] = attn_unpadding_idx
        model_inputs["ffn_padding_idx"] = ffn_padding_idx
        model_inputs["ffn_unpadding_idx"] = ffn_unpadding_idx

        return model_inputs, is_prefill

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
                            group_size=64)
            w2_config = PTQConfig(mode=quant_mode, backend=BackendTarget.ASCEND, weight_quant_dtype=msdtype.int8,
                                  act_quant_dtype=msdtype.int8, outliers_suppression=OutliersSuppressionType.SMOOTH)
            layer_policies = OrderedDict({r'.*\.feed_forward\.w2.*': w2_config,
                                          r'.*\.shared_experts.w2.*': w2_config})
        elif quant_type.lower() == 'smoothquant':
            cfg = PTQConfig(mode=quant_mode, backend=BackendTarget.ASCEND, weight_quant_dtype=msdtype.int8,
                            act_quant_dtype=msdtype.int8, outliers_suppression=OutliersSuppressionType.SMOOTH,
                            opname_blacklist=['lm_head', 'lkv2kv'])
            ffn_config = PTQConfig(mode=quant_mode, backend=BackendTarget.ASCEND, weight_quant_dtype=msdtype.int8,
                                   act_quant_dtype=msdtype.int8,
                                   outliers_suppression=OutliersSuppressionType.NONE,
                                   precision_recovery=PrecisionRecovery.NONE,
                                   act_quant_granularity=QuantGranularity.PER_TOKEN,
                                   weight_quant_granularity=QuantGranularity.PER_CHANNEL)
            layer_policies = OrderedDict({r'.*\.feed_forward\..*': ffn_config})
        elif quant_type.lower() == 'osl':
            cfg = PTQConfig(mode=quant_mode, backend=BackendTarget.ASCEND, weight_quant_dtype=msdtype.int8,
                            act_quant_dtype=msdtype.int8,
                            outliers_suppression=OutliersSuppressionType.OUTLIER_SUPPRESSION_LITE,
                            opname_blacklist=['lm_head', 'lkv2kv'])
            ffn_config = PTQConfig(mode=quant_mode, backend=BackendTarget.ASCEND, weight_quant_dtype=msdtype.int8,
                                   act_quant_dtype=msdtype.int8,
                                   outliers_suppression=OutliersSuppressionType.NONE,
                                   precision_recovery=PrecisionRecovery.NONE,
                                   act_quant_granularity=QuantGranularity.PER_TOKEN,
                                   weight_quant_granularity=QuantGranularity.PER_CHANNEL)
            layer_policies = OrderedDict({r'.*\.feed_forward\..*': ffn_config})
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
        if 'gptq-pergroup' in quant_type.lower():
            # pylint: disable=protected-access
            ptq.layer_policies[r'.*\.feed_forward\.w2.*'].aclnn_quant_list = ["w2"]
            ptq.layer_policies[r'.*\.shared_experts.w2.*'].aclnn_quant_list = ["w2"]
        ptq.decoder_layer_types.append(DeepseekV3DecodeLayer)
        return ptq
