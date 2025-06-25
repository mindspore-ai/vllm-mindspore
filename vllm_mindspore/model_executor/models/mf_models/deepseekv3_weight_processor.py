# Copyright 2025 Huawei Technologies Co., Ltd
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
"""
transform huggingface model to mindspore safetensor.
"""
import gc
import json
import os

import mindspore as ms
import numpy as np
from mindformers.parallel_core.inference.parallel_state import (
    get_tensor_model_parallel_rank)
from mindspore import dtype
from mindspore.communication.management import get_rank
from tqdm import tqdm
from vllm.logger import init_logger

from vllm_mindspore.model_executor.models.mf_models.weight_processor import (
    BaseWeightProcessor, EPMethod)

logger = init_logger(__name__)


def convert_np_to_ms_dtype(value):
    """convert_np_to_ms_dtype"""
    if value.dtype == np.int8:
        value_dtype = ms.int8
    elif value.dtype == np.int32:
        value_dtype = ms.int32
    elif value.dtype == np.int64:
        value_dtype = ms.int64
    elif value.dtype == np.float64:
        value_dtype = ms.float64
    elif value.dtype == np.float32:
        value_dtype = ms.float32
    else:
        value_dtype = ms.bfloat16
    return value_dtype


class DeepseekV3WeightProcessor(BaseWeightProcessor):
    r"""
    Provide DeepseekV3/R1 Model weight load and shards.
    Args:
        config (DeepseekV3/R1Config): The config of DeepseekV3/R1 model.
        network (InferenceDeepseekV3ForCausalLM): The network of DeepseekV3/R1.

    """

    def __init__(self, config, network, is_quant):
        super().__init__(config, network, is_quant)
        self.num_layers = self.config.model.model_config.num_layers
        self.expert_num = self.config.moe_config.expert_num
        self.moe_split_tp = self.moe_tp_size > 1
        self.moe_split_ep = self.moe_ep_size > 1
        logger.debug("Deepseekv3 weight split info:")
        logger.debug("global_rank_id: %s", self.global_rank_id)
        logger.debug("tp_group_size: %s", self.tp_group_size)
        logger.debug("dp_group_size: %s", self.dp_group_size)
        logger.debug("tp_rank_id: %s", self.tp_rank_id)
        logger.debug("ep_method: %s", self.ep_method.name)
        logger.debug("num_router_experts: %s", self.num_router_experts)
        logger.debug("ep_group_nums: %s", self.ep_group_nums)
        logger.debug("moe_ep_rank_id: %s", self.moe_ep_rank_id)
        logger.debug("moe_tp_rank_id: %s", self.moe_tp_rank_id)
        logger.debug("moe_ep_size: %s", self.moe_ep_size)
        logger.debug("moe_tp_size: %s", self.moe_tp_size)

    def quant_convert_weight_name(self, weight_name: str):
        """replace quant net weight name"""
        weight_name = weight_name.replace('embed_tokens.weight',
                                          'tok_embeddings.embedding_weight')

        weight_name = weight_name.replace('.self_attn.q_a_proj.weight',
                                          '.attention.q2l_proj._layer.weight')
        weight_name = weight_name.replace(
            '.self_attn.q_a_proj.input_scale',
            '.attention.q2l_proj.quant_op.input_scale')
        weight_name = weight_name.replace(
            '.self_attn.q_a_proj.input_offset',
            '.attention.q2l_proj.quant_op.input_zp')
        weight_name = weight_name.replace(
            '.self_attn.q_a_proj.quant_bias',
            '.attention.q2l_proj._layer.matmul.quant_bias')
        weight_name = weight_name.replace(
            '.self_attn.q_a_proj.deq_scale',
            '.attention.q2l_proj._layer.matmul.dequant_scale')

        weight_name = weight_name.replace('.self_attn.q_a_layernorm.weight',
                                          '.attention.lq_norm.weight')
        weight_name = weight_name.replace('.self_attn.kv_a_layernorm.weight',
                                          '.attention.lkv_norm.weight')
        weight_name = weight_name.replace('.self_attn.kv_b_proj.',
                                          '.attention.lkv2kv.')

        weight_name = weight_name.replace('.self_attn.q_b_proj.weight',
                                          '.attention.l2q_proj._layer.weight')
        weight_name = weight_name.replace(
            '.self_attn.q_b_proj.input_scale',
            '.attention.l2q_proj.quant_op.input_scale')
        weight_name = weight_name.replace(
            '.self_attn.q_b_proj.input_offset',
            '.attention.l2q_proj.quant_op.input_zp')
        weight_name = weight_name.replace(
            '.self_attn.q_b_proj.quant_bias',
            '.attention.l2q_proj._layer.matmul.quant_bias')
        weight_name = weight_name.replace(
            '.self_attn.q_b_proj.deq_scale',
            '.attention.l2q_proj._layer.matmul.dequant_scale')

        weight_name = weight_name.replace(
            '.self_attn.kv_a_proj_with_mqa.weight',
            '.attention.kv2l._layer.weight')
        weight_name = weight_name.replace(
            '.self_attn.kv_a_proj_with_mqa.input_scale',
            '.attention.kv2l.quant_op.input_scale')
        weight_name = weight_name.replace(
            '.self_attn.kv_a_proj_with_mqa.input_offset',
            '.attention.kv2l.quant_op.input_zp')
        weight_name = weight_name.replace(
            '.self_attn.kv_a_proj_with_mqa.quant_bias',
            '.attention.kv2l._layer.matmul.quant_bias')
        weight_name = weight_name.replace(
            '.self_attn.kv_a_proj_with_mqa.deq_scale',
            '.attention.kv2l._layer.matmul.dequant_scale')

        weight_name = weight_name.replace('.self_attn.o_proj.weight',
                                          '.attention.wo._layer.weight')
        weight_name = weight_name.replace(
            '.self_attn.o_proj.input_scale',
            '.attention.wo.quant_op.input_scale')
        weight_name = weight_name.replace('.self_attn.o_proj.input_offset',
                                          '.attention.wo.quant_op.input_zp')
        weight_name = weight_name.replace(
            '.self_attn.o_proj.quant_bias',
            '.attention.wo._layer.matmul.quant_bias')
        weight_name = weight_name.replace(
            '.self_attn.o_proj.deq_scale',
            '.attention.wo._layer.matmul.dequant_scale')

        weight_name = weight_name.replace('.self_attn.q_a_layernorm.bias',
                                          '.attention.l2q_proj.quant_op.beta')
        weight_name = weight_name.replace('.input_layernorm.bias',
                                          '.attention.q2l_proj.quant_op.beta')

        # mlp is pertoken quant
        weight_name = weight_name.replace('.weight_scale',
                                          '.matmul.weight_scale')
        weight_name = weight_name.replace('.weight_offset',
                                          '.matmul.weight_offset')

        weight_name = weight_name.replace('mlp.gate_proj.',
                                          'feed_forward.w1._layer.')
        weight_name = weight_name.replace('mlp.down_proj.',
                                          'feed_forward.w2._layer.')
        weight_name = weight_name.replace('mlp.up_proj.',
                                          'feed_forward.w3._layer.')
        weight_name = weight_name.replace('mlp.experts.',
                                          'feed_forward.routed_experts.ffn.')
        weight_name = weight_name.replace(
            'mlp.shared_experts.gate_proj.',
            'feed_forward.shared_experts.w1._layer.')
        weight_name = weight_name.replace(
            'mlp.shared_experts.down_proj.',
            'feed_forward.shared_experts.w2._layer.')
        weight_name = weight_name.replace(
            'mlp.shared_experts.up_proj.',
            'feed_forward.shared_experts.w3._layer.')
        weight_name = weight_name.replace(
            'mlp.gate.weight',
            'feed_forward.routed_experts.router.dense.weight')
        weight_name = weight_name.replace(
            'mlp.gate.e_score_correction_bias',
            'feed_forward.routed_experts.router.e_score_correction_bias')
        weight_name = weight_name.replace('.input_layernorm.weight',
                                          '.attention_norm.weight')
        weight_name = weight_name.replace('.post_attention_layernorm.',
                                          '.ffn_norm.')
        weight_name = weight_name.replace('model.norm.weight',
                                          'model.norm_out.weight')
        return weight_name

    def infer_trans_rope_weight(self, weight, qk_rope_head_dim):
        """process rope router weight"""
        w1 = weight[..., -qk_rope_head_dim::2, :]
        w2 = weight[..., -qk_rope_head_dim + 1::2, :]
        weight[..., -qk_rope_head_dim:, :] = np.concatenate([w1, w2], axis=-2)
        return weight

    def infer_quant_process_moe_with_ep(self, src_hf_dir, hf_weight_map,
                                        layer_id):
        w1_list = []
        w2_list = []
        w3_list = []

        w1_scale_list = []
        w2_scale_list = []
        w3_scale_list = []

        for index in range(self.ep_start, self.ep_stop):
            w1_hf_name = f"model.layers.{layer_id}.mlp.experts.{index}.gate_proj.weight"
            w2_hf_name = f"model.layers.{layer_id}.mlp.experts.{index}.down_proj.weight"
            w3_hf_name = f"model.layers.{layer_id}.mlp.experts.{index}.up_proj.weight"

            w1_ms_param, _ = self.get_safetensor_from_file(
                w1_hf_name, src_hf_dir, hf_weight_map)
            w2_ms_param, _ = self.get_safetensor_from_file(
                w2_hf_name, src_hf_dir, hf_weight_map)
            w3_ms_param, _ = self.get_safetensor_from_file(
                w3_hf_name, src_hf_dir, hf_weight_map)

            w1_list.append(w1_ms_param)
            w2_list.append(w2_ms_param)
            w3_list.append(w3_ms_param)

            w1_scale_hf_name = f"model.layers.{layer_id}.mlp.experts.{index}.gate_proj.weight_scale"
            w2_scale_hf_name = f"model.layers.{layer_id}.mlp.experts.{index}.down_proj.weight_scale"
            w3_scale_hf_name = f"model.layers.{layer_id}.mlp.experts.{index}.up_proj.weight_scale"

            w1_scale_ms_param, _ = self.get_safetensor_from_file(
                w1_scale_hf_name, src_hf_dir, hf_weight_map)
            w2_scale_ms_param, _ = self.get_safetensor_from_file(
                w2_scale_hf_name, src_hf_dir, hf_weight_map)
            w3_scale_ms_param, _ = self.get_safetensor_from_file(
                w3_scale_hf_name, src_hf_dir, hf_weight_map)

            w1_scale_ms_param = w1_scale_ms_param.squeeze(axis=-1)
            w2_scale_ms_param = w2_scale_ms_param.squeeze(axis=-1)
            w3_scale_ms_param = w3_scale_ms_param.squeeze(axis=-1)
            w1_scale_list.append(w1_scale_ms_param)
            w2_scale_list.append(w2_scale_ms_param)
            w3_scale_list.append(w3_scale_ms_param)

        return w1_list, w2_list, w3_list, w1_scale_list, w2_scale_list, w3_scale_list

    def infer_quant_process_moe_with_ep_tp(self, src_hf_dir, hf_weight_map,
                                           layer_id):
        w1_list = []
        w2_list = []
        w3_list = []

        w1_scale_list = []
        w2_scale_list = []
        w3_scale_list = []

        for index in range(self.ep_start, self.ep_stop):
            w1_hf_name = f"model.layers.{layer_id}.mlp.experts.{index}.gate_proj.weight"
            w2_hf_name = f"model.layers.{layer_id}.mlp.experts.{index}.down_proj.weight"
            w3_hf_name = f"model.layers.{layer_id}.mlp.experts.{index}.up_proj.weight"

            w1_ms_param, _ = self.get_safetensor_from_file_split_moe_tp_group(
                w1_hf_name, src_hf_dir, hf_weight_map, split_axis=0)
            w2_ms_param, _ = self.get_safetensor_from_file_split_moe_tp_group(
                w2_hf_name, src_hf_dir, hf_weight_map, split_axis=1)
            w3_ms_param, _ = self.get_safetensor_from_file_split_moe_tp_group(
                w3_hf_name, src_hf_dir, hf_weight_map, split_axis=0)

            w1_list.append(w1_ms_param)
            w2_list.append(w2_ms_param)
            w3_list.append(w3_ms_param)

            w1_scale_hf_name = f"model.layers.{layer_id}.mlp.experts.{index}.gate_proj.weight_scale"
            w2_scale_hf_name = f"model.layers.{layer_id}.mlp.experts.{index}.down_proj.weight_scale"
            w3_scale_hf_name = f"model.layers.{layer_id}.mlp.experts.{index}.up_proj.weight_scale"

            w1_scale_ms_param, _ = self.get_safetensor_from_file_split_moe_tp_group(
                w1_scale_hf_name, src_hf_dir, hf_weight_map, split_axis=0)
            w2_scale_ms_param, _ = self.get_safetensor_from_file(
                w2_scale_hf_name, src_hf_dir, hf_weight_map)
            w3_scale_ms_param, _ = self.get_safetensor_from_file_split_moe_tp_group(
                w3_scale_hf_name, src_hf_dir, hf_weight_map, split_axis=0)

            w1_scale_ms_param = w1_scale_ms_param.squeeze(axis=-1)
            w2_scale_ms_param = w2_scale_ms_param.squeeze(axis=-1)
            w3_scale_ms_param = w3_scale_ms_param.squeeze(axis=-1)
            w1_scale_list.append(w1_scale_ms_param)
            w2_scale_list.append(w2_scale_ms_param)
            w3_scale_list.append(w3_scale_ms_param)

        return w1_list, w2_list, w3_list, w1_scale_list, w2_scale_list, w3_scale_list

    def infer_quant_process_moe(self, src_hf_dir, hf_weight_map, layer_id):
        if self.moe_tp_size > 1:
            return self.infer_quant_process_moe_with_ep_tp(
                src_hf_dir, hf_weight_map, layer_id)
        else:
            return self.infer_quant_process_moe_with_ep(
                src_hf_dir, hf_weight_map, layer_id)

    def infer_quant_process_moe_routed_expert_ffn_weight(
            self, src_hf_dir, layer_id, hf_weight_map):
        """process moe router expert weight"""
        ffn_concat = self.config.model.model_config.ffn_concat

        # router expert dense
        router_dense_hf_name = f"model.layers.{layer_id}.mlp.gate.weight"
        router_dense_ms_name = self.quant_convert_weight_name(
            router_dense_hf_name)
        router_dense_ms_param, _ = self.get_safetensor_from_file(
            router_dense_hf_name, src_hf_dir, hf_weight_map)

        if self.moe_split_ep and self.ep_method != EPMethod.ALLTOALL:
            expert_idx = [idx for idx in range(router_dense_ms_param.shape[0])]
            in_start_expert_idx = self.ep_group_nums * self.moe_ep_rank_id
            expert_idx = expert_idx[
                in_start_expert_idx:] + expert_idx[:in_start_expert_idx]
            router_dense_ms_param = np.array(router_dense_ms_param)[expert_idx]

        self.parameter_dict[router_dense_ms_name] = ms.Parameter(
            ms.from_numpy(router_dense_ms_param).astype(ms.bfloat16),
            name=router_dense_ms_name,
            requires_grad=False)

        # e_score_correction_bias
        e_score_correction_bias_hf_name = f"model.layers.{layer_id}.mlp.gate.e_score_correction_bias"
        e_score_correction_bias_ms_name = self.quant_convert_weight_name(
            e_score_correction_bias_hf_name)
        e_score_correction_bias_ms_param, _ = self.get_safetensor_from_file(
            e_score_correction_bias_hf_name, src_hf_dir, hf_weight_map)
        self.parameter_dict[e_score_correction_bias_ms_name] = ms.Parameter(
            ms.from_numpy(e_score_correction_bias_ms_param).astype(ms.float32),
            name=e_score_correction_bias_ms_name,
            requires_grad=False)

        w1_ms_name = f"model.layers.{layer_id}.feed_forward.routed_experts.ffn.w1._layer.weight"
        w2_ms_name = f"model.layers.{layer_id}.feed_forward.routed_experts.ffn.w2._layer.weight"
        w3_ms_name = f"model.layers.{layer_id}.feed_forward.routed_experts.ffn.w3._layer.weight"

        w1_scale_ms_name = f"model.layers.{layer_id}.feed_forward.routed_experts.ffn.w1._layer.matmul.weight_scale"
        w2_scale_ms_name = f"model.layers.{layer_id}.feed_forward.routed_experts.ffn.w2._layer.matmul.weight_scale"
        w3_scale_ms_name = f"model.layers.{layer_id}.feed_forward.routed_experts.ffn.w3._layer.matmul.weight_scale"

        w1_list, w2_list, w3_list, w1_scale_list, w2_scale_list, w3_scale_list = \
            self.infer_quant_process_moe(src_hf_dir, hf_weight_map, layer_id)

        w1_ms_stack_param = np.stack(w1_list, axis=0)
        w2_ms_stack_param = np.stack(w2_list, axis=0)
        w3_ms_stack_param = np.stack(w3_list, axis=0)

        w1_scale_ms_stack_param = np.stack(w1_scale_list, axis=0)
        w2_scale_ms_stack_param = np.stack(w2_scale_list, axis=0)
        w3_scale_ms_stack_param = np.stack(w3_scale_list, axis=0)

        if ffn_concat:
            # w_gate_hidden
            w_gate_hidden_name = f"model.layers.{layer_id}.feed_forward.routed_experts.ffn.w_gate_hidden._layer.weight"
            w_gate_hidden_np = np.concatenate(
                [w1_ms_stack_param, w3_ms_stack_param], axis=1)
            w_gate_hidden_param = ms.from_numpy(w_gate_hidden_np).permute(
                0, 2, 1).astype(ms.int8)
            self.parameter_dict[w_gate_hidden_name] = ms.Parameter(
                w_gate_hidden_param,
                name=w_gate_hidden_name,
                requires_grad=False)
            # w_scale_gate_hidden
            w_scale_gate_hidden_name = \
                f"model.layers.{layer_id}.feed_forward.routed_experts.ffn.w_gate_hidden._layer.matmul.weight_scale"

            w_scale_gate_hidden_np = np.concatenate(
                [w1_scale_ms_stack_param, w3_scale_ms_stack_param], axis=1)
            w_scale_gate_hidden_param = ms.from_numpy(
                w_scale_gate_hidden_np).astype(ms.bfloat16)
            self.parameter_dict[w_scale_gate_hidden_name] = ms.Parameter(
                w_scale_gate_hidden_param,
                name=w_scale_gate_hidden_name,
                requires_grad=False)
        else:
            # w1 w3
            self.parameter_dict[w1_ms_name] = ms.Parameter(
                ms.from_numpy(w1_ms_stack_param).permute(0, 2,
                                                         1).astype(ms.int8),
                name=w1_ms_name,
                requires_grad=False)
            self.parameter_dict[w3_ms_name] = ms.Parameter(
                ms.from_numpy(w3_ms_stack_param).permute(0, 2,
                                                         1).astype(ms.int8),
                name=w3_ms_name,
                requires_grad=False)

            # w1_scale w3_scale
            self.parameter_dict[w1_scale_ms_name] = ms.Parameter(
                ms.from_numpy(w1_scale_ms_stack_param).astype(ms.bfloat16),
                name=w1_ms_name,
                requires_grad=False)
            self.parameter_dict[w3_scale_ms_name] = ms.Parameter(
                ms.from_numpy(w3_scale_ms_stack_param).astype(ms.bfloat16),
                name=w3_ms_name,
                requires_grad=False)

        self.parameter_dict[w2_ms_name] = ms.Parameter(
            ms.from_numpy(w2_ms_stack_param).permute(0, 2, 1).astype(ms.int8),
            name=w2_ms_name,
            requires_grad=False)

        self.parameter_dict[w2_scale_ms_name] = ms.Parameter(
            ms.from_numpy(w2_scale_ms_stack_param).astype(ms.bfloat16),
            name=w2_scale_ms_name,
            requires_grad=False)

    def get_quant_moe_shared_expert_weight(self, w1_hf_name, w2_hf_name,
                                           w3_hf_name, w1_scale_hf_name,
                                           w2_scale_hf_name, w3_scale_hf_name,
                                           src_hf_dir, hf_weight_map):
        if self.ep_method in [EPMethod.DEFAULT, EPMethod.ALLGATHER]:
            w1_ms_param, _ = self.get_safetensor_from_file_split_global_group(
                w1_hf_name, src_hf_dir, hf_weight_map, split_axis=0)
            w2_ms_param, _ = self.get_safetensor_from_file_split_global_group(
                w2_hf_name, src_hf_dir, hf_weight_map, split_axis=1)
            w3_ms_param, _ = self.get_safetensor_from_file_split_global_group(
                w3_hf_name, src_hf_dir, hf_weight_map, split_axis=0)
            w1_scale_ms_param, _ = self.get_safetensor_from_file_split_global_group(
                w1_scale_hf_name, src_hf_dir, hf_weight_map, split_axis=0)
            w2_scale_ms_param, _ = self.get_safetensor_from_file(
                w2_scale_hf_name, src_hf_dir, hf_weight_map)

            w3_scale_ms_param, _ = self.get_safetensor_from_file_split_global_group(
                w3_scale_hf_name, src_hf_dir, hf_weight_map, split_axis=0)
        elif self.ep_method == EPMethod.ALLTOALL:
            w1_ms_param, _ = self.get_safetensor_from_file(
                w1_hf_name, src_hf_dir, hf_weight_map)
            w2_ms_param, _ = self.get_safetensor_from_file(
                w2_hf_name, src_hf_dir, hf_weight_map)
            w3_ms_param, _ = self.get_safetensor_from_file(
                w3_hf_name, src_hf_dir, hf_weight_map)

            w1_scale_ms_param, _ = self.get_safetensor_from_file(
                w1_scale_hf_name, src_hf_dir, hf_weight_map)
            w2_scale_ms_param, _ = self.get_safetensor_from_file(
                w2_scale_hf_name, src_hf_dir, hf_weight_map)
            w3_scale_ms_param, _ = self.get_safetensor_from_file(
                w3_scale_hf_name, src_hf_dir, hf_weight_map)
        else:
            raise ValueError("Unsupported ep_method: %s", self.ep_method)

        return w1_ms_param, w2_ms_param, w3_ms_param, w1_scale_ms_param, w2_scale_ms_param, w3_scale_ms_param

    def infer_quant_process_moe_shared_expert_ffn_weight(
            self, src_hf_dir, layer_id, hf_weight_map):
        """infer quant process moe shared expert ffn weight"""
        ffn_concat = self.config.model.model_config.ffn_concat
        w1_hf_name = f"model.layers.{layer_id}.mlp.shared_experts.gate_proj.weight"
        w2_hf_name = f"model.layers.{layer_id}.mlp.shared_experts.down_proj.weight"
        w3_hf_name = f"model.layers.{layer_id}.mlp.shared_experts.up_proj.weight"

        w1_scale_hf_name = f"model.layers.{layer_id}.mlp.shared_experts.gate_proj.weight_scale"
        w2_scale_hf_name = f"model.layers.{layer_id}.mlp.shared_experts.down_proj.weight_scale"
        w3_scale_hf_name = f"model.layers.{layer_id}.mlp.shared_experts.up_proj.weight_scale"

        w1_ms_name = self.quant_convert_weight_name(w1_hf_name)
        w2_ms_name = self.quant_convert_weight_name(w2_hf_name)
        w3_ms_name = self.quant_convert_weight_name(w3_hf_name)

        w1_scale_ms_name = self.quant_convert_weight_name(w1_scale_hf_name)
        w2_scale_ms_name = self.quant_convert_weight_name(w2_scale_hf_name)
        w3_scale_ms_name = self.quant_convert_weight_name(w3_scale_hf_name)

        w1_ms_param, w2_ms_param, w3_ms_param, w1_scale_ms_param, w2_scale_ms_param, w3_scale_ms_param = \
            self.get_quant_moe_shared_expert_weight(w1_hf_name, w2_hf_name, w3_hf_name, w1_scale_hf_name,
                                                    w2_scale_hf_name,
                                                    w3_scale_hf_name, src_hf_dir, hf_weight_map)

        w1_scale_ms_param = w1_scale_ms_param.squeeze(axis=-1)
        w2_scale_ms_param = w2_scale_ms_param.squeeze(axis=-1)
        w3_scale_ms_param = w3_scale_ms_param.squeeze(axis=-1)

        if ffn_concat:
            w_gate_hidden_name = f"model.layers.{layer_id}.feed_forward.shared_experts.w_gate_hidden._layer.weight"
            w_gate_hidden_np = np.concatenate([w1_ms_param, w3_ms_param],
                                              axis=0)
            w_gate_hidden_param = ms.from_numpy(w_gate_hidden_np).astype(
                ms.int8)
            self.parameter_dict[w_gate_hidden_name] = ms.Parameter(
                w_gate_hidden_param,
                name=w_gate_hidden_name,
                requires_grad=False)

            w_scale_gate_hidden_name = \
                f"model.layers.{layer_id}.feed_forward.shared_experts.w_gate_hidden._layer.matmul.weight_scale"
            w_scale_gate_hidden_np = np.concatenate(
                [w1_scale_ms_param, w3_scale_ms_param], axis=0)
            w_scale_gate_hidden_param = ms.from_numpy(
                w_scale_gate_hidden_np).astype(ms.bfloat16)
            self.parameter_dict[w_scale_gate_hidden_name] = ms.Parameter(
                w_scale_gate_hidden_param,
                name=w_scale_gate_hidden_name,
                requires_grad=False)

        else:
            self.parameter_dict[w1_ms_name] = ms.Parameter(
                ms.from_numpy(w1_ms_param).astype(ms.int8),
                name=w1_ms_name,
                requires_grad=False)
            self.parameter_dict[w3_ms_name] = ms.Parameter(
                ms.from_numpy(w3_ms_param).astype(ms.int8),
                name=w3_ms_name,
                requires_grad=False)

            self.parameter_dict[w1_scale_ms_name] = ms.Parameter(
                ms.from_numpy(w1_scale_ms_param).astype(ms.bfloat16),
                name=w1_ms_name,
                requires_grad=False)
            self.parameter_dict[w3_scale_ms_name] = ms.Parameter(
                ms.from_numpy(w3_scale_ms_param).astype(ms.bfloat16),
                name=w3_ms_name,
                requires_grad=False)

        self.parameter_dict[w2_ms_name] = ms.Parameter(
            ms.from_numpy(w2_ms_param).astype(ms.int8),
            name=w2_ms_name,
            requires_grad=False)

        self.parameter_dict[w2_scale_ms_name] = ms.Parameter(
            ms.from_numpy(w2_scale_ms_param).astype(ms.bfloat16),
            name=w2_ms_name,
            requires_grad=False)

    def infer_quant_process_dense_ffn_weight(self, src_hf_dir, layer_id,
                                             hf_weight_map):
        """infer process dense ffn weight"""

        ffn_concat = self.config.model.model_config.ffn_concat
        w1_hf_name = f"model.layers.{layer_id}.mlp.gate_proj.weight"
        w1_ms_name = self.quant_convert_weight_name(w1_hf_name)
        w1_ms_param, _ = self.get_safetensor_from_file_split_tp_group(
            w1_hf_name, src_hf_dir, hf_weight_map, split_axis=0)
        w1_scale_hf_name = f"model.layers.{layer_id}.mlp.gate_proj.weight_scale"
        w1_scale_ms_name = self.quant_convert_weight_name(w1_scale_hf_name)
        w1_scale_ms_param, _ = self.get_safetensor_from_file_split_tp_group(
            w1_scale_hf_name, src_hf_dir, hf_weight_map, split_axis=0)

        w2_hf_name = f"model.layers.{layer_id}.mlp.down_proj.weight"
        w2_ms_name = self.quant_convert_weight_name(w2_hf_name)
        w2_ms_param, _ = self.get_safetensor_from_file_split_tp_group(
            w2_hf_name, src_hf_dir, hf_weight_map, split_axis=1)
        w2_scale_hf_name = f"model.layers.{layer_id}.mlp.down_proj.weight_scale"
        w2_scale_ms_name = self.quant_convert_weight_name(w2_scale_hf_name)
        # shape:[7168,1]
        w2_scale_ms_param, _ = self.get_safetensor_from_file(
            w2_scale_hf_name, src_hf_dir, hf_weight_map)

        w3_hf_name = f"model.layers.{layer_id}.mlp.up_proj.weight"
        w3_ms_name = self.quant_convert_weight_name(w3_hf_name)
        w3_ms_param, _ = self.get_safetensor_from_file_split_tp_group(
            w3_hf_name, src_hf_dir, hf_weight_map, split_axis=0)
        w3_scale_hf_name = f"model.layers.{layer_id}.mlp.up_proj.weight_scale"
        w3_scale_ms_name = self.quant_convert_weight_name(w3_scale_hf_name)
        w3_scale_ms_param, _ = self.get_safetensor_from_file_split_tp_group(
            w3_scale_hf_name, src_hf_dir, hf_weight_map, split_axis=0)

        w1_scale_ms_param = w1_scale_ms_param.squeeze(axis=-1)
        w2_scale_ms_param = w2_scale_ms_param.squeeze(axis=-1)
        w3_scale_ms_param = w3_scale_ms_param.squeeze(axis=-1)

        if ffn_concat:
            w_gate_hidden_name = f"model.layers.{layer_id}.feed_forward.w_gate_hidden._layer.weight"
            w_gate_hidden_np = np.concatenate([w1_ms_param, w3_ms_param],
                                              axis=0)
            w_gate_hidden_param = ms.from_numpy(w_gate_hidden_np).astype(
                dtype=ms.int8)
            self.parameter_dict[w_gate_hidden_name] = ms.Parameter(
                w_gate_hidden_param,
                name=w_gate_hidden_name,
                requires_grad=False)

            w_scale_gate_hidden_name = f"model.layers.{layer_id}.feed_forward.w_gate_hidden._layer.matmul.weight_scale"
            w_scale_gate_hidden_param = ms.from_numpy(
                np.concatenate([w1_scale_ms_param, w3_scale_ms_param],
                               axis=0)).astype(dtype=ms.bfloat16)
            self.parameter_dict[w_scale_gate_hidden_name] = ms.Parameter(
                w_scale_gate_hidden_param,
                name=w_scale_gate_hidden_name,
                requires_grad=False)

        else:
            self.parameter_dict[w1_ms_name] = ms.Parameter(
                ms.from_numpy(w1_ms_param).astype(ms.int8),
                name=w1_ms_name,
                requires_grad=False)
            self.parameter_dict[w3_ms_name] = ms.Parameter(
                ms.from_numpy(w3_ms_param).astype(ms.int8),
                name=w3_ms_name,
                requires_grad=False)

            self.parameter_dict[w1_scale_ms_name] = ms.Parameter(
                ms.from_numpy(w1_scale_ms_param).astype(ms.bfloat16),
                name=w1_scale_ms_name,
                requires_grad=False)
            self.parameter_dict[w3_scale_ms_name] = ms.Parameter(
                ms.from_numpy(w3_scale_ms_param).astype(ms.bfloat16),
                name=w3_scale_ms_name,
                requires_grad=False)

        self.parameter_dict[w2_ms_name] = ms.Parameter(
            ms.from_numpy(w2_ms_param).astype(ms.int8),
            name=w2_ms_name,
            requires_grad=False)

        self.parameter_dict[w2_scale_ms_name] = ms.Parameter(
            ms.from_numpy(w2_scale_ms_param).astype(ms.bfloat16),
            name=w2_ms_name,
            requires_grad=False)

    def infer_convert_outer_weight(self, src_hf_dir, hf_weight_map):
        """convert weight not in model"""
        embed_tokens_hf_name = "model.embed_tokens.weight"
        embed_tokens_ms_name = self.quant_convert_weight_name(
            embed_tokens_hf_name)
        np_data, _ = self.get_safetensor_from_file(embed_tokens_hf_name,
                                                   src_hf_dir, hf_weight_map)
        self.parameter_dict[embed_tokens_ms_name] = ms.Parameter(
            ms.from_numpy(np_data).astype(ms.bfloat16),
            name=embed_tokens_ms_name,
            requires_grad=False)

        norm_hf_name = "model.norm.weight"
        norm_ms_name = self.quant_convert_weight_name(norm_hf_name)
        np_data, _ = self.get_safetensor_from_file(norm_hf_name, src_hf_dir,
                                                   hf_weight_map)
        self.parameter_dict[norm_ms_name] = ms.Parameter(
            ms.from_numpy(np_data).astype(ms.bfloat16),
            name=norm_ms_name,
            requires_grad=False)

        lm_head_hf_name = "lm_head.weight"
        lm_head_ms_name = self.quant_convert_weight_name(lm_head_hf_name)
        if not self.config.parallel_config.vocab_emb_dp:
            np_data, _ = self.get_safetensor_from_file_split_tp_group(
                lm_head_hf_name, src_hf_dir, hf_weight_map, split_axis=0)
        else:
            np_data, _ = self.get_safetensor_from_file(lm_head_hf_name,
                                                       src_hf_dir,
                                                       hf_weight_map)
        self.parameter_dict[lm_head_ms_name] = ms.Parameter(
            ms.from_numpy(np_data).astype(ms.bfloat16),
            name=lm_head_ms_name,
            requires_grad=False)

    def quant_special_attention_weight(self,
                                       layer_id,
                                       src_hf_dir,
                                       hf_weight_map,
                                       name,
                                       is_trans_rope_weigh=False,
                                       is_split_param=False):
        # q_a_proj->q2l_proj
        # kv_a_proj_with_mqa->kv2l
        # q_a_layernorm->lq_norm
        # o_proj->wo

        # input_scale, input_zp no split
        input_scale_hf_name = f"model.layers.{layer_id}.self_attn." + name + ".input_scale"
        input_scale_ms_name = self.quant_convert_weight_name(
            input_scale_hf_name)
        input_scale_ms_param, _ = self.get_safetensor_from_file(
            input_scale_hf_name, src_hf_dir, hf_weight_map)
        self.parameter_dict[input_scale_ms_name] = ms.Parameter(
            ms.from_numpy(input_scale_ms_param).astype(ms.bfloat16),
            name=input_scale_ms_name,
            requires_grad=False)

        input_zp_hf_name = f"model.layers.{layer_id}.self_attn." + name + ".input_offset"
        input_zp_ms_name = self.quant_convert_weight_name(input_zp_hf_name)
        input_zp_ms_param, _ = self.get_safetensor_from_file(
            input_zp_hf_name, src_hf_dir, hf_weight_map)
        self.parameter_dict[input_zp_ms_name] = ms.Parameter(
            ms.from_numpy(input_zp_ms_param).astype(ms.int8),
            name=input_zp_ms_name,
            requires_grad=False)

        if not is_trans_rope_weigh:
            quant_bias_hf_name = f"model.layers.{layer_id}.self_attn." + name + ".quant_bias"
            quant_bias_ms_name = self.quant_convert_weight_name(
                quant_bias_hf_name)
            quant_bias_ms_param, _ = self.get_safetensor_from_file(
                quant_bias_hf_name, src_hf_dir, hf_weight_map)
            if name == "o_proj" and self.tp_rank_id != 0:
                quant_bias_ms_param.fill(0)

            dequant_scale_hf_name = f"model.layers.{layer_id}.self_attn." + name + ".deq_scale"
            dequant_scale_ms_name = self.quant_convert_weight_name(
                dequant_scale_hf_name)
            dequant_scale_ms_param, _ = self.get_safetensor_from_file(
                dequant_scale_hf_name, src_hf_dir, hf_weight_map)
        else:
            kv_lora_rank = self.config.model.model_config.kv_lora_rank
            qk_rope_head_dim = self.config.model.model_config.qk_rope_head_dim
            qk_nope_head_dim = self.config.model.model_config.qk_nope_head_dim

            num_heads = self.config.model.model_config.num_heads
            rope_dim = qk_rope_head_dim + qk_nope_head_dim
            kv_head_dim = kv_lora_rank + qk_rope_head_dim

            quant_bias_hf_name = f"model.layers.{layer_id}.self_attn." + name + ".quant_bias"
            quant_bias_ms_name = self.quant_convert_weight_name(
                quant_bias_hf_name)
            quant_bias_ms_param, _ = self.get_safetensor_from_file(
                quant_bias_hf_name, src_hf_dir, hf_weight_map)

            dequant_scale_hf_name = f"model.layers.{layer_id}.self_attn." + name + ".deq_scale"
            dequant_scale_ms_name = self.quant_convert_weight_name(
                dequant_scale_hf_name)
            dequant_scale_ms_param, _ = self.get_safetensor_from_file(
                dequant_scale_hf_name, src_hf_dir, hf_weight_map)

            if name == "q_b_proj":
                quant_bias_ms_param = quant_bias_ms_param.reshape(
                    num_heads, rope_dim, -1)
                quant_bias_ms_param = self.infer_trans_rope_weight(
                    quant_bias_ms_param, qk_rope_head_dim)
                quant_bias_ms_param = quant_bias_ms_param.reshape(
                    num_heads * rope_dim, -1).reshape(-1)

                dequant_scale_ms_param = dequant_scale_ms_param.reshape(
                    num_heads, rope_dim, -1)
                dequant_scale_ms_param = self.infer_trans_rope_weight(
                    dequant_scale_ms_param, qk_rope_head_dim)
                dequant_scale_ms_param = dequant_scale_ms_param.reshape(
                    num_heads * rope_dim, -1).reshape(-1)

            elif name == "kv_a_proj_with_mqa":
                quant_bias_ms_param = quant_bias_ms_param.reshape(
                    kv_head_dim, -1)
                quant_bias_ms_param = self.infer_trans_rope_weight(
                    quant_bias_ms_param, qk_rope_head_dim).reshape(-1)

                dequant_scale_ms_param = dequant_scale_ms_param.reshape(
                    kv_head_dim, -1)
                dequant_scale_ms_param = self.infer_trans_rope_weight(
                    dequant_scale_ms_param, qk_rope_head_dim).reshape(-1)

        if is_split_param:
            quant_bias_ms_param = self.split_weight_by_rank(
                quant_bias_ms_param, split_axis=0)
            dequant_scale_ms_param = self.split_weight_by_rank(
                dequant_scale_ms_param, split_axis=0)

        self.parameter_dict[quant_bias_ms_name] = ms.Parameter(
            ms.from_numpy(quant_bias_ms_param).astype(ms.int32),
            name=quant_bias_ms_name,
            requires_grad=False)
        self.parameter_dict[dequant_scale_ms_name] = ms.Parameter(
            ms.from_numpy(dequant_scale_ms_param).astype(ms.float32),
            name=dequant_scale_ms_name,
            requires_grad=False)

    def infer_quant_bias_weight(self, src_hf_dir, layer_id, hf_weight_map):
        # quant_op.beta
        q2l_proj_bias_hf_name = f"model.layers.{layer_id}.input_layernorm.bias"
        q2l_proj_bias_ms_name = self.quant_convert_weight_name(
            q2l_proj_bias_hf_name)
        q2l_proj_bias_ms_param, _ = self.get_safetensor_from_file(
            q2l_proj_bias_hf_name, src_hf_dir, hf_weight_map)

        kv2l_bias_ms_name = f"model.layers.{layer_id}.attention.kv2l.quant_op.beta"
        kv2l_bias_ms_param = q2l_proj_bias_ms_param.copy()

        l2q_proj_bias_hf_name = f"model.layers.{layer_id}.self_attn.q_a_layernorm.bias"
        l2q_proj_bias_ms_name = self.quant_convert_weight_name(
            l2q_proj_bias_hf_name)
        l2q_proj_bias_ms_param, _ = self.get_safetensor_from_file(
            l2q_proj_bias_hf_name, src_hf_dir, hf_weight_map)

        self.parameter_dict[q2l_proj_bias_ms_name] = ms.Parameter(
            ms.from_numpy(q2l_proj_bias_ms_param).astype(ms.bfloat16),
            name=q2l_proj_bias_ms_name,
            requires_grad=False)
        self.parameter_dict[kv2l_bias_ms_name] = ms.Parameter(
            ms.from_numpy(kv2l_bias_ms_param).astype(ms.bfloat16),
            name=kv2l_bias_ms_name,
            requires_grad=False)
        self.parameter_dict[l2q_proj_bias_ms_name] = ms.Parameter(
            ms.from_numpy(l2q_proj_bias_ms_param).astype(ms.bfloat16),
            name=l2q_proj_bias_ms_name,
            requires_grad=False)

    def infer_quant_process_attention_weight(self, src_hf_dir, layer_id,
                                             hf_weight_map):
        """infer quant process attention weight"""
        num_heads = self.config.model.model_config.num_heads
        qk_rope_head_dim = self.config.model.model_config.qk_rope_head_dim
        v_head_dim = self.config.model.model_config.v_head_dim
        qk_nope_head_dim = self.config.model.model_config.qk_nope_head_dim

        rope_dim = qk_rope_head_dim + qk_nope_head_dim

        # q_a_layernorm->lq_norm
        lq_norm_hf_name = f"model.layers.{layer_id}.self_attn.q_a_layernorm.weight"
        lq_norm_ms_name = self.quant_convert_weight_name(lq_norm_hf_name)
        lq_norm_ms_param, _ = self.get_safetensor_from_file(
            lq_norm_hf_name, src_hf_dir, hf_weight_map)
        self.parameter_dict[lq_norm_ms_name] = ms.Parameter(
            ms.from_numpy(lq_norm_ms_param).astype(ms.bfloat16),
            name=lq_norm_ms_name,
            requires_grad=False)

        # q_b_proj->l2q_proj
        l2q_proj_hf_name = f"model.layers.{layer_id}.self_attn.q_b_proj.weight"
        l2q_proj_ms_name = self.quant_convert_weight_name(l2q_proj_hf_name)
        l2q_proj_ms_param, _ = self.get_safetensor_from_file(
            l2q_proj_hf_name, src_hf_dir, hf_weight_map)
        l2q_proj_ms_param = l2q_proj_ms_param.reshape(num_heads, rope_dim, -1)
        l2q_proj_ms_param = self.infer_trans_rope_weight(
            l2q_proj_ms_param, qk_rope_head_dim)
        l2q_proj_ms_param = l2q_proj_ms_param.reshape(num_heads * rope_dim, -1)
        l2q_proj_ms_param = self.split_weight_by_rank(l2q_proj_ms_param,
                                                      split_axis=0)
        self.parameter_dict[l2q_proj_ms_name] = ms.Parameter(
            ms.from_numpy(l2q_proj_ms_param).astype(ms.int8),
            name=l2q_proj_ms_name,
            requires_grad=False)
        self.quant_special_attention_weight(layer_id,
                                            src_hf_dir,
                                            hf_weight_map,
                                            "q_b_proj",
                                            is_trans_rope_weigh=True,
                                            is_split_param=True)

        # kv_a_layernorm->lkv_norm
        lkv_norm_hf_name = f"model.layers.{layer_id}.self_attn.kv_a_layernorm.weight"
        lkv_norm_ms_name = self.quant_convert_weight_name(lkv_norm_hf_name)
        lkv_norm_ms_param, _ = self.get_safetensor_from_file(
            lkv_norm_hf_name, src_hf_dir, hf_weight_map)
        self.parameter_dict[lkv_norm_ms_name] = ms.Parameter(
            ms.from_numpy(lkv_norm_ms_param).astype(ms.bfloat16),
            name=lkv_norm_ms_name,
            requires_grad=False)

        # kv_b_proj->lkv2kv
        lkv2kv_hf_name = f"model.layers.{layer_id}.self_attn.kv_b_proj.weight"
        lkv2kv_ms_name = self.quant_convert_weight_name(lkv2kv_hf_name)
        lkv2kv_ms_param, _ = self.get_safetensor_from_file(
            lkv2kv_hf_name, src_hf_dir, hf_weight_map)
        lkv2kv_head = qk_nope_head_dim + v_head_dim
        lkv2kv_ms_param = lkv2kv_ms_param.reshape(num_heads, lkv2kv_head, -1)
        value_k_nope, value_v = lkv2kv_ms_param[:, :
                                                qk_nope_head_dim, :], lkv2kv_ms_param[:,
                                                                                      qk_nope_head_dim:, :]

        # value_k_nope
        value_k_nope = value_k_nope.reshape(-1, value_k_nope.shape[-1])
        value_k_nope = self.split_weight_by_rank(value_k_nope, split_axis=0)
        name_k_nope = lkv2kv_ms_name.replace(".attention.lkv2kv.",
                                             ".attention.lkv2kv_k_nope.")
        self.parameter_dict[name_k_nope] = ms.Parameter(
            ms.from_numpy(value_k_nope).astype(ms.bfloat16),
            name=name_k_nope,
            requires_grad=False)
        # value_v
        value_v = value_v.reshape(-1, value_v.shape[-1])
        value_v = self.split_weight_by_rank(value_v, split_axis=0)
        name_v = lkv2kv_ms_name.replace(".attention.lkv2kv.",
                                        ".attention.lkv2kv_v.")
        self.parameter_dict[name_v] = ms.Parameter(
            ms.from_numpy(value_v).astype(ms.bfloat16),
            name=name_v,
            requires_grad=False)

        # o_proj->wo
        wo_hf_name = f"model.layers.{layer_id}.self_attn.o_proj.weight"
        wo_ms_name = self.quant_convert_weight_name(wo_hf_name)
        wo_ms_param, _ = self.get_safetensor_from_file(wo_hf_name, src_hf_dir,
                                                       hf_weight_map)
        wo_ms_param = self.split_weight_by_rank(wo_ms_param, split_axis=1)
        self.parameter_dict[wo_ms_name] = ms.Parameter(
            ms.from_numpy(wo_ms_param).astype(ms.int8),
            name=wo_ms_name,
            requires_grad=False)
        self.quant_special_attention_weight(layer_id, src_hf_dir,
                                            hf_weight_map, "o_proj")

    def infer_quant_process_dense_qkv_weight(self, src_hf_dir, layer_id,
                                             hf_weight_map):
        """infer_quant_process_dense_qkv_weight"""
        parameter_dict = {}
        kv_lora_rank = self.config.model.model_config.kv_lora_rank
        qk_rope_head_dim = self.config.model.model_config.qk_rope_head_dim
        kv_head_dim = kv_lora_rank + qk_rope_head_dim

        qkv_concat = self.config.model.model_config.qkv_concat
        # q2l
        q2l_hf_name = f"model.layers.{layer_id}.self_attn.q_a_proj.weight"
        q2l_ms_name = self.quant_convert_weight_name(q2l_hf_name)
        q2l_ms_param, _ = self.get_safetensor_from_file(
            q2l_hf_name, src_hf_dir, hf_weight_map)

        q2l_input_scale_hf_name = f"model.layers.{layer_id}.self_attn.q_a_proj.input_scale"
        q2l_input_scale_ms_name = self.quant_convert_weight_name(
            q2l_input_scale_hf_name)
        q2l_input_scale_ms_param, _ = self.get_safetensor_from_file(
            q2l_input_scale_hf_name, src_hf_dir, hf_weight_map)

        q2l_input_zp_hf_name = f"model.layers.{layer_id}.self_attn.q_a_proj.input_offset"
        q2l_input_zp_ms_name = self.quant_convert_weight_name(
            q2l_input_zp_hf_name)
        q2l_input_zp_ms_param, _ = self.get_safetensor_from_file(
            q2l_input_zp_hf_name, src_hf_dir, hf_weight_map)

        q2l_quant_bias_hf_name = f"model.layers.{layer_id}.self_attn.q_a_proj.quant_bias"
        q2l_quant_bias_ms_name = self.quant_convert_weight_name(
            q2l_quant_bias_hf_name)
        q2l_quant_bias_ms_param, _ = self.get_safetensor_from_file(
            q2l_quant_bias_hf_name, src_hf_dir, hf_weight_map)

        q2l_dequant_scale_hf_name = f"model.layers.{layer_id}.self_attn.q_a_proj.deq_scale"
        q2l_dequant_scale_ms_name = self.quant_convert_weight_name(
            q2l_dequant_scale_hf_name)
        q2l_dequant_scale_ms_param, _ = self.get_safetensor_from_file(
            q2l_dequant_scale_hf_name, src_hf_dir, hf_weight_map)
        # kv2l
        kv2l_hf_name = f"model.layers.{layer_id}.self_attn.kv_a_proj_with_mqa.weight"
        kv2l_ms_name = self.quant_convert_weight_name(kv2l_hf_name)
        kv2l_ms_param, _ = self.get_safetensor_from_file(
            kv2l_hf_name, src_hf_dir, hf_weight_map)
        kv2l_ms_param = kv2l_ms_param.reshape(kv_head_dim, -1)
        kv2l_ms_param = self.infer_trans_rope_weight(kv2l_ms_param,
                                                     qk_rope_head_dim)

        kv2l_input_scale_hf_name = f"model.layers.{layer_id}.self_attn.kv_a_proj_with_mqa.input_scale"
        kv2l_input_scale_ms_name = self.quant_convert_weight_name(
            kv2l_input_scale_hf_name)
        kv2l_input_scale_ms_param, _ = self.get_safetensor_from_file(
            kv2l_input_scale_hf_name, src_hf_dir, hf_weight_map)

        kv2l_input_zp_hf_name = f"model.layers.{layer_id}.self_attn.kv_a_proj_with_mqa.input_offset"
        kv2l_input_zp_ms_name = self.quant_convert_weight_name(
            kv2l_input_zp_hf_name)
        kv2l_input_zp_ms_param, _ = self.get_safetensor_from_file(
            kv2l_input_zp_hf_name, src_hf_dir, hf_weight_map)

        kv2l_quant_bias_hf_name = f"model.layers.{layer_id}.self_attn.kv_a_proj_with_mqa.quant_bias"
        kv2l_quant_bias_ms_name = self.quant_convert_weight_name(
            kv2l_quant_bias_hf_name)
        kv2l_quant_bias_ms_param, _ = self.get_safetensor_from_file(
            kv2l_quant_bias_hf_name, src_hf_dir, hf_weight_map)
        kv2l_quant_bias_ms_param = kv2l_quant_bias_ms_param.reshape(
            kv_head_dim, -1)
        kv2l_quant_bias_ms_param = self.infer_trans_rope_weight(
            kv2l_quant_bias_ms_param, qk_rope_head_dim).reshape(-1)

        kv2l_dequant_scale_hf_name = f"model.layers.{layer_id}.self_attn.kv_a_proj_with_mqa.deq_scale"
        kv2l_dequant_scale_ms_name = self.quant_convert_weight_name(
            kv2l_dequant_scale_hf_name)
        kv2l_dequant_scale_ms_param, _ = self.get_safetensor_from_file(
            kv2l_dequant_scale_hf_name, src_hf_dir, hf_weight_map)
        kv2l_dequant_scale_ms_param = kv2l_dequant_scale_ms_param.reshape(
            kv_head_dim, -1)
        kv2l_dequant_scale_ms_param = self.infer_trans_rope_weight(
            kv2l_dequant_scale_ms_param, qk_rope_head_dim).reshape(-1)

        attn_rmsnorm_beta_hf_name = f"model.layers.{layer_id}.input_layernorm.bias"
        attn_rmsnorm_beta_ms_name = self.quant_convert_weight_name(
            attn_rmsnorm_beta_hf_name)
        attn_rmsnorm_beta_ms_param, _ = self.get_safetensor_from_file(
            attn_rmsnorm_beta_hf_name, src_hf_dir, hf_weight_map)

        if qkv_concat:
            qkv2l_weight_name = f"model.layers.{layer_id}.attention.qkv2l._layer.weight"
            qkv2l_bias_name = f"model.layers.{layer_id}.attention.qkv2l._layer.matmul.quant_bias"
            qkv2l_scale_name = f"model.layers.{layer_id}.attention.qkv2l._layer.matmul.dequant_scale"
            qkv2l_quant_zp_name = f"model.layers.{layer_id}.attention.qkv2l.quant_op.input_zp"
            qkv2l_quant_scale_name = f"model.layers.{layer_id}.attention.qkv2l.quant_op.input_scale"
            qkv2l_rmsnorm_beta_name = f"model.layers.{layer_id}.attention.qkv2l.quant_op.beta"

            qkv2l_weight = np.concatenate((q2l_ms_param, kv2l_ms_param), 0)
            parameter_dict[qkv2l_weight_name] = ms.Parameter(
                ms.Tensor(qkv2l_weight, ms.int8),
                name=qkv2l_weight_name,
                requires_grad=False)
            qkv2l_bias = np.concatenate(
                (q2l_quant_bias_ms_param, kv2l_quant_bias_ms_param), 0)
            parameter_dict[qkv2l_bias_name] = ms.Parameter(
                ms.Tensor(qkv2l_bias, ms.int32),
                name=qkv2l_bias_name,
                requires_grad=False)
            qkv2l_scale = np.concatenate(
                (q2l_dequant_scale_ms_param, kv2l_dequant_scale_ms_param), 0)
            parameter_dict[qkv2l_scale_name] = ms.Parameter(
                ms.Tensor(qkv2l_scale, ms.float32),
                name=qkv2l_scale_name,
                requires_grad=False)
            parameter_dict[qkv2l_quant_zp_name] = ms.Parameter(
                ms.Tensor(q2l_input_zp_ms_param, ms.int8), requires_grad=False)
            parameter_dict[qkv2l_quant_scale_name] = ms.Parameter(
                ms.Tensor(q2l_input_scale_ms_param, ms.bfloat16),
                requires_grad=False)
            parameter_dict[qkv2l_rmsnorm_beta_name] = ms.Parameter(
                ms.Tensor(attn_rmsnorm_beta_ms_param, ms.float32),
                requires_grad=False)
        else:
            parameter_dict[q2l_ms_name] = ms.Parameter(ms.Tensor(
                q2l_ms_param, ms.int8),
                                                       name=q2l_ms_name,
                                                       requires_grad=False)
            parameter_dict[kv2l_ms_name] = ms.Parameter(ms.Tensor(
                kv2l_ms_param, ms.int8),
                                                        requires_grad=False)
            parameter_dict[q2l_quant_bias_ms_name] = ms.Parameter(
                ms.Tensor(q2l_quant_bias_ms_param, ms.int32),
                name=q2l_quant_bias_ms_name,
                requires_grad=False)
            parameter_dict[kv2l_quant_bias_ms_name] = ms.Parameter(
                ms.Tensor(kv2l_quant_bias_ms_param, ms.int32),
                name=kv2l_quant_bias_ms_name,
                requires_grad=False)
            parameter_dict[q2l_dequant_scale_ms_name] = ms.Parameter(
                ms.Tensor(q2l_dequant_scale_ms_param, ms.float32),
                name=q2l_dequant_scale_ms_name,
                requires_grad=False)
            parameter_dict[kv2l_dequant_scale_ms_name] = ms.Parameter(
                ms.Tensor(kv2l_dequant_scale_ms_param, ms.float32),
                name=kv2l_dequant_scale_ms_name,
                requires_grad=False)
            parameter_dict[q2l_input_zp_ms_name] = ms.Parameter(
                ms.Tensor(q2l_input_zp_ms_param, ms.int8),
                name=q2l_input_zp_ms_name,
                requires_grad=False)
            parameter_dict[kv2l_input_zp_ms_name] = ms.Parameter(
                ms.Tensor(kv2l_input_zp_ms_param, ms.int8),
                name=kv2l_input_zp_ms_name,
                requires_grad=False)
            parameter_dict[q2l_input_scale_ms_name] = ms.Parameter(
                ms.Tensor(q2l_input_scale_ms_param, ms.bfloat16),
                name=q2l_input_scale_ms_name,
                requires_grad=False)
            parameter_dict[kv2l_input_scale_ms_name] = ms.Parameter(
                ms.Tensor(kv2l_input_scale_ms_param, ms.bfloat16),
                name=kv2l_input_scale_ms_name,
                requires_grad=False)
            parameter_dict[attn_rmsnorm_beta_ms_name] = ms.Parameter(
                ms.Tensor(attn_rmsnorm_beta_ms_param, ms.float32),
                name=attn_rmsnorm_beta_ms_name,
                requires_grad=False)
        _, _ = ms.load_param_into_net(self.network, parameter_dict)
        del parameter_dict
        gc.collect()

    def infer_quant_net_convert_layer_weight(self, src_hf_dir, layer_id,
                                             hf_weight_map):
        """infer quant net convert layer weight"""

        if layer_id >= 3:
            self.infer_quant_process_moe_routed_expert_ffn_weight(
                src_hf_dir, layer_id, hf_weight_map)
            self.infer_quant_process_moe_shared_expert_ffn_weight(
                src_hf_dir, layer_id, hf_weight_map)
        else:
            self.infer_quant_process_dense_ffn_weight(src_hf_dir, layer_id,
                                                      hf_weight_map)

        self.infer_quant_process_dense_qkv_weight(src_hf_dir, layer_id,
                                                  hf_weight_map)
        self.infer_quant_process_attention_weight(src_hf_dir, layer_id,
                                                  hf_weight_map)
        self.infer_quant_bias_weight(src_hf_dir, layer_id, hf_weight_map)
        self.infer_process_norm_weight(src_hf_dir, layer_id, hf_weight_map)

    def convert_weight_name(self, weight_name: str):
        """replace weight name"""
        weight_name = weight_name.replace('embed_tokens.weight',
                                          'tok_embeddings.embedding_weight')
        weight_name = weight_name.replace('.self_attn.q_a_proj.',
                                          '.attention.q2l_proj.')
        weight_name = weight_name.replace('.self_attn.q_a_layernorm.',
                                          '.attention.lq_norm.')
        weight_name = weight_name.replace('.self_attn.q_b_proj.',
                                          '.attention.l2q_proj.')
        weight_name = weight_name.replace('.self_attn.kv_a_proj_with_mqa.',
                                          '.attention.kv2l.')
        weight_name = weight_name.replace('.self_attn.kv_a_layernorm.',
                                          '.attention.lkv_norm.')
        weight_name = weight_name.replace('.self_attn.kv_b_proj.',
                                          '.attention.lkv2kv.')
        weight_name = weight_name.replace('.self_attn.o_proj.',
                                          '.attention.wo.')
        weight_name = weight_name.replace('mlp.gate_proj.', 'feed_forward.w1.')
        weight_name = weight_name.replace('mlp.down_proj.', 'feed_forward.w2.')
        weight_name = weight_name.replace('mlp.up_proj.', 'feed_forward.w3.')
        weight_name = weight_name.replace('mlp.experts.',
                                          'feed_forward.routed_experts.ffn.')
        weight_name = weight_name.replace('mlp.shared_experts.gate_proj.',
                                          'feed_forward.shared_experts.w1.')
        weight_name = weight_name.replace('mlp.shared_experts.down_proj.',
                                          'feed_forward.shared_experts.w2.')
        weight_name = weight_name.replace('mlp.shared_experts.up_proj.',
                                          'feed_forward.shared_experts.w3.')
        weight_name = weight_name.replace(
            'mlp.gate.weight',
            'feed_forward.routed_experts.router.dense.weight')
        weight_name = weight_name.replace(
            'mlp.gate.e_score_correction_bias',
            'feed_forward.routed_experts.router.e_score_correction_bias')
        weight_name = weight_name.replace('.input_layernorm.',
                                          '.attention_norm.')
        weight_name = weight_name.replace('.post_attention_layernorm.',
                                          '.ffn_norm.')
        weight_name = weight_name.replace('model.norm.weight',
                                          'model.norm_out.weight')

        weight_name = self.convert_mtp_weight_name(weight_name)
        return weight_name

    def convert_mtp_weight_name(self, weight_name: str):
        layer = 0 if 'layers.' not in weight_name else int(
            weight_name[weight_name.find('layers.'):].split('.')[1])
        if layer < self.num_layers:
            return weight_name
        mtp_prefix = 'mtp_model'
        is_mtp_layer = 'tok_embeddings' not in weight_name and 'shared_head.' not in weight_name
        mtp_prefix = mtp_prefix if not is_mtp_layer else f'{mtp_prefix}.layer'
        is_decode_layer = "ffn" in weight_name or "attention" in weight_name or "feed_forward" in weight_name
        mtp_prefix = mtp_prefix if not is_decode_layer else f'{mtp_prefix}.decode_layer'

        weight_name = weight_name.replace(f'model.layers.{layer}', mtp_prefix)
        if "tok_embeddings" in weight_name:
            weight_name = weight_name.replace('.weight', '.embedding_weight')
        if "shared_head." in weight_name:
            weight_name = weight_name.replace('shared_head.', '')
        return weight_name

    def infer_process_moe_routed_expert_ffn_weight(self, src_hf_dir, layer_id,
                                                   hf_weight_map):
        """process moe router expert weight"""
        ffn_concat = self.config.model.model_config.ffn_concat

        # router expert dense
        router_dense_hf_name = f"model.layers.{layer_id}.mlp.gate.weight"
        router_dense_ms_name = self.convert_weight_name(router_dense_hf_name)
        router_dense_ms_param, _ = self.get_safetensor_from_file(
            router_dense_hf_name, src_hf_dir, hf_weight_map)
        self.parameter_dict[router_dense_ms_name] = ms.Parameter(
            ms.from_numpy(router_dense_ms_param).astype(ms.bfloat16),
            name=router_dense_ms_name,
            requires_grad=False)

        # e_score_correction_bias
        e_score_correction_bias_hf_name = f"model.layers.{layer_id}.mlp.gate.e_score_correction_bias"
        e_score_correction_bias_ms_name = self.convert_weight_name(
            e_score_correction_bias_hf_name)
        e_score_correction_bias_ms_param, _ = self.get_safetensor_from_file(
            e_score_correction_bias_hf_name, src_hf_dir, hf_weight_map)
        self.parameter_dict[e_score_correction_bias_ms_name] = ms.Parameter(
            ms.from_numpy(e_score_correction_bias_ms_param).astype(ms.float32),
            name=e_score_correction_bias_ms_name,
            requires_grad=False)

        w1_list = []
        w2_list = []
        w3_list = []

        w1_ms_name = f"model.layers.{layer_id}.feed_forward.routed_experts.ffn.w1.weight"
        w1_ms_name = w1_ms_name if layer_id < self.num_layers else self.convert_mtp_weight_name(
            w1_ms_name)
        w2_ms_name = f"model.layers.{layer_id}.feed_forward.routed_experts.ffn.w2.weight"
        w2_ms_name = w2_ms_name if layer_id < self.num_layers else self.convert_mtp_weight_name(
            w2_ms_name)
        w3_ms_name = f"model.layers.{layer_id}.feed_forward.routed_experts.ffn.w3.weight"
        w3_ms_name = w3_ms_name if layer_id < self.num_layers else self.convert_mtp_weight_name(
            w3_ms_name)

        for index in range(0, self.num_router_experts):
            w1_hf_name = f"model.layers.{layer_id}.mlp.experts.{index}.gate_proj.weight"
            w1_ms_param, _ = self.get_safetensor_from_file_split_tp_group(
                w1_hf_name, src_hf_dir, hf_weight_map, split_axis=0)

            w2_hf_name = f"model.layers.{layer_id}.mlp.experts.{index}.down_proj.weight"
            w2_ms_param, _ = self.get_safetensor_from_file_split_tp_group(
                w2_hf_name, src_hf_dir, hf_weight_map, split_axis=1)

            w3_hf_name = f"model.layers.{layer_id}.mlp.experts.{index}.up_proj.weight"
            w3_ms_param, _ = self.get_safetensor_from_file_split_tp_group(
                w3_hf_name, src_hf_dir, hf_weight_map, split_axis=0)

            w1_list.append(w1_ms_param)
            w2_list.append(w2_ms_param)
            w3_list.append(w3_ms_param)

        w1_ms_stack_param = np.stack(w1_list, axis=0)
        w2_ms_stack_param = np.stack(w2_list, axis=0)
        w3_ms_stack_param = np.stack(w3_list, axis=0)

        if ffn_concat:
            w_gate_hidden_name = f"model.layers.{layer_id}.feed_forward.routed_experts.ffn.w_gate_hidden.weight"
            w_gate_hidden_name = w_gate_hidden_name if layer_id < self.num_layers else \
                self.convert_mtp_weight_name(w_gate_hidden_name)
            w_gate_hidden_np = np.concatenate(
                [w1_ms_stack_param, w3_ms_stack_param], axis=1)
            w_gate_hidden_param = ms.from_numpy(w_gate_hidden_np).permute(
                0, 2, 1).astype(dtype=ms.bfloat16)
            self.parameter_dict[w_gate_hidden_name] = ms.Parameter(
                w_gate_hidden_param,
                name=w_gate_hidden_name,
                requires_grad=False)
        else:
            w1_ms_stack_param = ms.from_numpy(w1_ms_stack_param).permute(
                0, 2, 1).astype(ms.bfloat16)
            self.parameter_dict[w1_ms_name] = ms.Parameter(w1_ms_stack_param,
                                                           name=w1_ms_name,
                                                           requires_grad=False)

            w3_ms_stack_param = ms.from_numpy(w3_ms_stack_param).permute(
                0, 2, 1).astype(ms.bfloat16)
            self.parameter_dict[w3_ms_name] = ms.Parameter(w3_ms_stack_param,
                                                           name=w3_ms_name,
                                                           requires_grad=False)

        w2_ms_stack_param = ms.from_numpy(w2_ms_stack_param).permute(
            0, 2, 1).astype(ms.bfloat16)
        self.parameter_dict[w2_ms_name] = ms.Parameter(w2_ms_stack_param,
                                                       name=w2_ms_name,
                                                       requires_grad=False)

    def get_moe_shared_expert_weight(self, w1_hf_name, w2_hf_name, w3_hf_name,
                                     src_hf_dir, hf_weight_map):
        if self.ep_method in [EPMethod.DEFAULT, EPMethod.ALLGATHER]:
            w1_ms_param, _ = self.get_safetensor_from_file_split_global_group(
                w1_hf_name, src_hf_dir, hf_weight_map, split_axis=0)
            w2_ms_param, _ = self.get_safetensor_from_file_split_global_group(
                w2_hf_name, src_hf_dir, hf_weight_map, split_axis=1)
            w3_ms_param, _ = self.get_safetensor_from_file_split_global_group(
                w3_hf_name, src_hf_dir, hf_weight_map, split_axis=0)
        elif self.ep_method == EPMethod.ALLTOALL:
            w1_ms_param, _ = self.get_safetensor_from_file(
                w1_hf_name, src_hf_dir, hf_weight_map)
            w2_ms_param, _ = self.get_safetensor_from_file(
                w2_hf_name, src_hf_dir, hf_weight_map)
            w3_ms_param, _ = self.get_safetensor_from_file(
                w3_hf_name, src_hf_dir, hf_weight_map)

        else:
            raise ValueError("Unsupported ep_method: %s", self.ep_method)

        return w1_ms_param, w2_ms_param, w3_ms_param

    def infer_process_moe_shared_expert_ffn_weight(self, src_hf_dir, layer_id,
                                                   hf_weight_map):
        """infer process moe shared expert ffn weight"""
        ffn_concat = self.config.model.model_config.ffn_concat
        w1_hf_name = f"model.layers.{layer_id}.mlp.shared_experts.gate_proj.weight"
        w2_hf_name = f"model.layers.{layer_id}.mlp.shared_experts.down_proj.weight"
        w3_hf_name = f"model.layers.{layer_id}.mlp.shared_experts.up_proj.weight"

        w1_ms_name = self.convert_weight_name(w1_hf_name)
        w2_ms_name = self.convert_weight_name(w2_hf_name)
        w3_ms_name = self.convert_weight_name(w3_hf_name)

        w1_ms_param, w2_ms_param, w3_ms_param = self.get_moe_shared_expert_weight(
            w1_hf_name, w2_hf_name, w3_hf_name, src_hf_dir, hf_weight_map)

        if ffn_concat:
            w_gate_hidden_name = f"model.layers.{layer_id}.feed_forward.shared_experts.w_gate_hidden.weight"
            w_gate_hidden_name = w_gate_hidden_name if layer_id < self.num_layers else \
                self.convert_mtp_weight_name(w_gate_hidden_name)
            w_gate_hidden_np = np.concatenate([w1_ms_param, w3_ms_param],
                                              axis=0)
            w_gate_hidden_param = ms.from_numpy(w_gate_hidden_np).astype(
                ms.bfloat16)
            self.parameter_dict[w_gate_hidden_name] = ms.Parameter(
                w_gate_hidden_param,
                name=w_gate_hidden_name,
                requires_grad=False)
        else:
            self.parameter_dict[w1_ms_name] = ms.Parameter(
                ms.from_numpy(w1_ms_param).astype(ms.bfloat16),
                name=w1_ms_name,
                requires_grad=False)
            self.parameter_dict[w3_ms_name] = ms.Parameter(
                ms.from_numpy(w3_ms_param).astype(ms.bfloat16),
                name=w3_ms_name,
                requires_grad=False)
        self.parameter_dict[w2_ms_name] = ms.Parameter(
            ms.from_numpy(w2_ms_param).astype(ms.bfloat16),
            name=w2_ms_name,
            requires_grad=False)

    def infer_process_dense_ffn_weight(self, src_hf_dir, layer_id,
                                       hf_weight_map):
        """infer process dense ffn weight"""

        ffn_concat = self.config.model.model_config.ffn_concat

        w1_hf_name = f"model.layers.{layer_id}.mlp.gate_proj.weight"
        w1_ms_name = self.convert_weight_name(w1_hf_name)
        w1_ms_param, _ = self.get_safetensor_from_file_split_tp_group(
            w1_hf_name, src_hf_dir, hf_weight_map, split_axis=0)

        w2_hf_name = f"model.layers.{layer_id}.mlp.down_proj.weight"
        w2_ms_name = self.convert_weight_name(w2_hf_name)
        w2_ms_param, _ = self.get_safetensor_from_file_split_tp_group(
            w2_hf_name, src_hf_dir, hf_weight_map, split_axis=1)

        w3_hf_name = f"model.layers.{layer_id}.mlp.up_proj.weight"
        w3_ms_name = self.convert_weight_name(w3_hf_name)
        w3_ms_param, _ = self.get_safetensor_from_file_split_tp_group(
            w3_hf_name, src_hf_dir, hf_weight_map, split_axis=0)

        if ffn_concat:
            w_gate_hidden_name = f"model.layers.{layer_id}.feed_forward.w_gate_hidden.weight"
            w_gate_hidden_np = np.concatenate([w1_ms_param, w3_ms_param],
                                              axis=0)
            w_gate_hidden_param = ms.from_numpy(w_gate_hidden_np).astype(
                ms.bfloat16)
            self.parameter_dict[w_gate_hidden_name] = ms.Parameter(
                w_gate_hidden_param,
                name=w_gate_hidden_name,
                requires_grad=False)
        else:
            self.parameter_dict[w1_ms_name] = ms.Parameter(
                ms.from_numpy(w1_ms_param).astype(ms.bfloat16),
                name=w1_ms_name,
                requires_grad=False)
            self.parameter_dict[w3_ms_name] = ms.Parameter(
                ms.from_numpy(w3_ms_param).astype(ms.bfloat16),
                name=w3_ms_name,
                requires_grad=False)

        self.parameter_dict[w2_ms_name] = ms.Parameter(
            ms.from_numpy(w2_ms_param).astype(ms.bfloat16),
            name=w2_ms_name,
            requires_grad=False)

    def infer_process_attention_weight(self, src_hf_dir, layer_id,
                                       hf_weight_map):
        """infer process attention weight"""
        num_heads = self.config.model.model_config.num_heads
        kv_lora_rank = self.config.model.model_config.kv_lora_rank
        qk_rope_head_dim = self.config.model.model_config.qk_rope_head_dim
        v_head_dim = self.config.model.model_config.v_head_dim
        qk_nope_head_dim = self.config.model.model_config.qk_nope_head_dim

        rope_dim = qk_rope_head_dim + qk_nope_head_dim
        kv_head_dim = kv_lora_rank + qk_rope_head_dim

        qkv_concat = self.config.model.model_config.qkv_concat
        # q2l_proj
        q2l_proj_hf_name = f"model.layers.{layer_id}.self_attn.q_a_proj.weight"
        q2l_proj_ms_name = self.convert_weight_name(q2l_proj_hf_name)
        q_a_proj_ms_param, _ = self.get_safetensor_from_file(
            q2l_proj_hf_name, src_hf_dir, hf_weight_map)

        # kv2l
        kv2l_hf_name = f"model.layers.{layer_id}.self_attn.kv_a_proj_with_mqa.weight"
        kv2l_ms_name = self.convert_weight_name(kv2l_hf_name)
        kv2l_ms_param, _ = self.get_safetensor_from_file(
            kv2l_hf_name, src_hf_dir, hf_weight_map)
        kv2l_ms_param = kv2l_ms_param.reshape(kv_head_dim, -1)
        kv2l_ms_param = self.infer_trans_rope_weight(kv2l_ms_param,
                                                     qk_rope_head_dim)
        if qkv_concat:
            wqkv2l_weight = np.concatenate((q_a_proj_ms_param, kv2l_ms_param),
                                           0)
            wqkv2l_weight_name = f"model.layers.{layer_id}.attention.qkv2l.weight"
            self.parameter_dict[wqkv2l_weight_name] = ms.Parameter(
                ms.from_numpy(wqkv2l_weight).astype(ms.bfloat16),
                name=wqkv2l_weight_name,
                requires_grad=False)
        else:
            self.parameter_dict[q2l_proj_ms_name] = ms.Parameter(
                ms.from_numpy(q_a_proj_ms_param).astype(ms.bfloat16),
                name=q2l_proj_ms_name,
                requires_grad=False)
            self.parameter_dict[kv2l_ms_name] = ms.Parameter(
                ms.from_numpy(kv2l_ms_param).astype(ms.bfloat16),
                name=kv2l_ms_name,
                requires_grad=False)
        # lq_norm
        lq_norm_hf_name = f"model.layers.{layer_id}.self_attn.q_a_layernorm.weight"
        lq_norm_ms_name = self.convert_weight_name(lq_norm_hf_name)
        lq_norm_ms_param, _ = self.get_safetensor_from_file(
            lq_norm_hf_name, src_hf_dir, hf_weight_map)
        self.parameter_dict[lq_norm_ms_name] = ms.Parameter(
            ms.from_numpy(lq_norm_ms_param).astype(ms.bfloat16),
            name=lq_norm_ms_name,
            requires_grad=False)

        # l2q_proj
        l2q_proj_hf_name = f"model.layers.{layer_id}.self_attn.q_b_proj.weight"
        l2q_proj_ms_name = self.convert_weight_name(l2q_proj_hf_name)
        l2q_proj_ms_param, _ = self.get_safetensor_from_file(
            l2q_proj_hf_name, src_hf_dir, hf_weight_map)
        l2q_proj_ms_param = l2q_proj_ms_param.reshape(num_heads, rope_dim, -1)
        l2q_proj_ms_param = self.infer_trans_rope_weight(
            l2q_proj_ms_param, qk_rope_head_dim)
        l2q_proj_ms_param = l2q_proj_ms_param.reshape(num_heads * rope_dim, -1)
        l2q_proj_ms_param = self.split_weight_by_rank(l2q_proj_ms_param,
                                                      split_axis=0)
        self.parameter_dict[l2q_proj_ms_name] = ms.Parameter(
            ms.from_numpy(l2q_proj_ms_param).astype(ms.bfloat16),
            name=l2q_proj_ms_name,
            requires_grad=False)

        # lkv_norm
        lkv_norm_hf_name = f"model.layers.{layer_id}.self_attn.kv_a_layernorm.weight"
        lkv_norm_ms_name = self.convert_weight_name(lkv_norm_hf_name)
        lkv_norm_ms_param, _ = self.get_safetensor_from_file(
            lkv_norm_hf_name, src_hf_dir, hf_weight_map)
        self.parameter_dict[lkv_norm_ms_name] = ms.Parameter(
            ms.from_numpy(lkv_norm_ms_param).astype(ms.bfloat16),
            name=lkv_norm_ms_name,
            requires_grad=False)

        # lkv2kv
        lkv2kv_hf_name = f"model.layers.{layer_id}.self_attn.kv_b_proj.weight"
        lkv2kv_ms_name = self.convert_weight_name(lkv2kv_hf_name)
        lkv2kv_ms_param, _ = self.get_safetensor_from_file(
            lkv2kv_hf_name, src_hf_dir, hf_weight_map)
        lkv2kv_head = qk_nope_head_dim + v_head_dim
        lkv2kv_ms_param = lkv2kv_ms_param.reshape(num_heads, lkv2kv_head, -1)
        value_k_nope, value_v = lkv2kv_ms_param[:, :
                                                qk_nope_head_dim, :], lkv2kv_ms_param[:,
                                                                                      qk_nope_head_dim:, :]

        # value_k_nope
        value_k_nope = value_k_nope.reshape(-1, value_k_nope.shape[-1])
        value_k_nope = self.split_weight_by_rank(value_k_nope, split_axis=0)
        name_k_nope = lkv2kv_ms_name.replace(".attention.lkv2kv.",
                                             ".attention.lkv2kv_k_nope.")
        self.parameter_dict[name_k_nope] = ms.Parameter(
            ms.from_numpy(value_k_nope).astype(ms.bfloat16),
            name=name_k_nope,
            requires_grad=False)
        # value_v
        value_v = value_v.reshape(-1, value_v.shape[-1])
        value_v = self.split_weight_by_rank(value_v, split_axis=0)
        name_v = lkv2kv_ms_name.replace(".attention.lkv2kv.",
                                        ".attention.lkv2kv_v.")
        self.parameter_dict[name_v] = ms.Parameter(
            ms.from_numpy(value_v).astype(ms.bfloat16),
            name=name_v,
            requires_grad=False)

        # wo
        wo_hf_name = f"model.layers.{layer_id}.self_attn.o_proj.weight"
        wo_ms_name = self.convert_weight_name(wo_hf_name)
        wo_ms_param, _ = self.get_safetensor_from_file(wo_hf_name, src_hf_dir,
                                                       hf_weight_map)
        wo_ms_param = self.split_weight_by_rank(wo_ms_param, split_axis=1)
        self.parameter_dict[wo_ms_name] = ms.Parameter(
            ms.from_numpy(wo_ms_param).astype(ms.bfloat16),
            name=wo_ms_name,
            requires_grad=False)

    def infer_process_norm_weight(self, src_hf_dir, layer_id, hf_weight_map):
        """infer process attention weight"""
        # attention_norm
        attention_norm_hf_name = f"model.layers.{layer_id}.input_layernorm.weight"
        attention_norm_ms_name = self.convert_weight_name(
            attention_norm_hf_name)
        attention_norm_ms_param, _ = self.get_safetensor_from_file(
            attention_norm_hf_name, src_hf_dir, hf_weight_map)
        self.parameter_dict[attention_norm_ms_name] = ms.Parameter(
            ms.from_numpy(attention_norm_ms_param).astype(ms.bfloat16),
            name=attention_norm_ms_name,
            requires_grad=False)

        # ffn_norm
        ffn_norm_hf_name = f"model.layers.{layer_id}.post_attention_layernorm.weight"
        ffn_norm_ms_name = self.convert_weight_name(ffn_norm_hf_name)
        ffn_norm_ms_param, _ = self.get_safetensor_from_file(
            ffn_norm_hf_name, src_hf_dir, hf_weight_map)
        self.parameter_dict[ffn_norm_ms_name] = ms.Parameter(
            ms.from_numpy(ffn_norm_ms_param).astype(ms.bfloat16),
            name=ffn_norm_ms_name,
            requires_grad=False)

    def infer_process_mtp_layer_weight(self, src_hf_dir, layer_id,
                                       hf_weight_map):
        parameter_dict = {}
        mtp_layer_names = [
            "embed_tokens.weight", "enorm.weight", "hnorm.weight",
            "eh_proj.weight", "shared_head.norm.weight",
            "shared_head.head.weight"
        ]
        head_names = ["eh_proj.weight", "shared_head.head.weight"]
        for prefix_name in mtp_layer_names:
            hf_name = f"model.layers.{layer_id}.{prefix_name}"
            ms_name = self.convert_weight_name(hf_name)
            if prefix_name in head_names and not self.config.parallel_config.vocab_emb_dp:
                ms_param, _ = self.get_safetensor_from_file_split_tp_group(
                    hf_name, src_hf_dir, hf_weight_map, split_axis=0)
            else:
                ms_param, _ = self.get_safetensor_from_file(
                    hf_name, src_hf_dir, hf_weight_map)
            parameter_dict[ms_name] = ms.Parameter(ms.Tensor(
                ms_param, ms.bfloat16),
                                                   name=ms_name,
                                                   requires_grad=False)

        _, ckpt_not_load = ms.load_param_into_net(self.network, parameter_dict)

    def infer_convert_layer_weight(self, src_hf_dir, layer_id, hf_weight_map):
        """infer convert layer weight"""
        if layer_id >= 3:
            self.infer_process_moe_routed_expert_ffn_weight(
                src_hf_dir, layer_id, hf_weight_map)
            self.infer_process_moe_shared_expert_ffn_weight(
                src_hf_dir, layer_id, hf_weight_map)
        else:
            self.infer_process_dense_ffn_weight(src_hf_dir, layer_id,
                                                hf_weight_map)

        self.infer_process_attention_weight(src_hf_dir, layer_id,
                                            hf_weight_map)
        self.infer_process_norm_weight(src_hf_dir, layer_id, hf_weight_map)

        # convert mtp shared weights.
        if layer_id >= self.num_layers:
            self.infer_process_mtp_layer_weight(src_hf_dir, layer_id,
                                                hf_weight_map)

    def smooth_quant_process_route_ffn_weight(self, src_hf_dir, layer_id,
                                              hf_weight_map, parameter_dict,
                                              layer_type):
        """smooth_quant_process_route_ffn_weight"""

        ffn_concat = self.config.model.model_config.ffn_concat
        w1_weight_name = f"model.layers.{layer_id}.{layer_type}.w1._layer.weight"
        w1_scale_name = f"model.layers.{layer_id}.{layer_type}.w1._layer.matmul.weight_scale"
        w3_weight_name = f"model.layers.{layer_id}.{layer_type}.w3._layer.weight"
        w3_scale_name = f"model.layers.{layer_id}.{layer_type}.w3._layer.matmul.weight_scale"
        w2_weight_name = f"model.layers.{layer_id}.{layer_type}.w2._layer.weight"
        w2_scale_name = f"model.layers.{layer_id}.{layer_type}.w2._layer.matmul.weight_scale"
        w1_weight_param, _ = self.get_routed_safetensor_3_dim(
            w1_weight_name,
            src_hf_dir,
            hf_weight_map,
            tp_axis=2,
            split_ep=self.moe_split_ep,
            split_tp=self.moe_split_tp)

        w1_scale_param, _ = self.get_routed_safetensor_2_dim(
            w1_scale_name,
            src_hf_dir,
            hf_weight_map,
            tp_axis=1,
            split_ep=self.moe_split_ep,
            split_tp=self.moe_split_tp)

        w3_weight_param, _ = self.get_routed_safetensor_3_dim(
            w3_weight_name,
            src_hf_dir,
            hf_weight_map,
            tp_axis=2,
            split_ep=self.moe_split_ep,
            split_tp=self.moe_split_tp)

        w3_scale_param, _ = self.get_routed_safetensor_2_dim(
            w3_scale_name,
            src_hf_dir,
            hf_weight_map,
            tp_axis=1,
            split_ep=self.moe_split_ep,
            split_tp=self.moe_split_tp)

        w2_weight_param, _ = self.get_routed_safetensor_3_dim(
            w2_weight_name,
            src_hf_dir,
            hf_weight_map,
            tp_axis=1,
            split_ep=self.moe_split_ep,
            split_tp=self.moe_split_tp)
        w2_scale_param, _ = self.get_routed_safetensor_2_dim(
            w2_scale_name,
            src_hf_dir,
            hf_weight_map,
            split_ep=self.moe_split_ep,
            split_tp=False)
        if ffn_concat:
            concat_weight_name = f"model.layers.{layer_id}.{layer_type}.w_gate_hidden._layer.weight"
            concat_weight_param = ms.Tensor(np.concatenate(
                [w1_weight_param, w3_weight_param], axis=2),
                                            dtype=ms.int8)
            parameter_dict[concat_weight_name] = ms.Parameter(
                concat_weight_param,
                name=concat_weight_name,
                requires_grad=False)

            concat_scale_name = f"model.layers.{layer_id}.{layer_type}.w_gate_hidden._layer.matmul.weight_scale"
            concat_scale_param = ms.Tensor(np.concatenate(
                [w1_scale_param, w3_scale_param], axis=1),
                                           dtype=ms.bfloat16)
            parameter_dict[concat_scale_name] = ms.Parameter(
                concat_scale_param,
                name=concat_scale_name,
                requires_grad=False)
        else:
            # w1 w3
            parameter_dict[w1_weight_name] = ms.Parameter(ms.Tensor(
                w1_weight_param, ms.int8),
                                                          name=w1_weight_name,
                                                          requires_grad=False)
            parameter_dict[w3_weight_name] = ms.Parameter(ms.Tensor(
                w3_weight_param, ms.int8),
                                                          name=w3_weight_name,
                                                          requires_grad=False)

            parameter_dict[w1_scale_name] = ms.Parameter(ms.Tensor(
                w1_scale_param, ms.bfloat16),
                                                         name=w1_scale_name,
                                                         requires_grad=False)
            parameter_dict[w3_scale_name] = ms.Parameter(ms.Tensor(
                w3_scale_param, ms.bfloat16),
                                                         name=w3_scale_name,
                                                         requires_grad=False)

        parameter_dict[w2_weight_name] = ms.Parameter(ms.Tensor(
            w2_weight_param, ms.int8),
                                                      name=w2_weight_name,
                                                      requires_grad=False)
        parameter_dict[w2_scale_name] = ms.Parameter(ms.Tensor(
            w2_scale_param, ms.bfloat16),
                                                     name=w2_scale_name,
                                                     requires_grad=False)

    def get_smooth_quant_moe_shared_expert_weight(
            self, w1_weight_name, w1_scale_name, w3_weight_name, w3_scale_name,
            w2_weight_name, src_hf_dir, hf_weight_map):
        '''get_smooth_quant_moe_shared_expert_weight'''

        if self.ep_method in [EPMethod.DEFAULT, EPMethod.ALLGATHER]:
            w1_weight_param, _ = self.get_safetensor_from_file_split_moe_tp_group(
                w1_weight_name, src_hf_dir, hf_weight_map, split_axis=0)

            w1_scale_param, _ = self.get_safetensor_from_file_split_moe_tp_group(
                w1_scale_name, src_hf_dir, hf_weight_map, split_axis=0)

            w3_weight_param, _ = self.get_safetensor_from_file_split_moe_tp_group(
                w3_weight_name, src_hf_dir, hf_weight_map, split_axis=0)
            w3_scale_param, _ = self.get_safetensor_from_file_split_moe_tp_group(
                w3_scale_name, src_hf_dir, hf_weight_map, split_axis=0)

            w2_weight_param, _ = self.get_safetensor_from_file_split_moe_tp_group(
                w2_weight_name, src_hf_dir, hf_weight_map, split_axis=1)
        elif self.ep_method == EPMethod.ALLTOALL:
            w1_weight_param, _ = self.get_safetensor_from_file(
                w1_weight_name, src_hf_dir, hf_weight_map)
            w1_scale_param, _ = self.get_safetensor_from_file(
                w1_scale_name, src_hf_dir, hf_weight_map)

            w3_weight_param, _ = self.get_safetensor_from_file(
                w3_weight_name, src_hf_dir, hf_weight_map)
            w3_scale_param, _ = self.get_safetensor_from_file(
                w3_scale_name, src_hf_dir, hf_weight_map)

            w2_weight_param, _ = self.get_safetensor_from_file(
                w2_weight_name, src_hf_dir, hf_weight_map)
        else:
            raise ValueError("Unsupported ep_method: %s", self.ep_method)

        return w1_weight_param, w1_scale_param, w3_weight_param, w3_scale_param, w2_weight_param

    def smooth_quant_process_shared_ffn_weight(self, src_hf_dir, layer_id,
                                               hf_weight_map, parameter_dict,
                                               layer_type):
        """smooth_quant_process_shared_ffn_weight"""

        ffn_concat = self.config.model.model_config.ffn_concat
        w1_weight_name = f"model.layers.{layer_id}.{layer_type}.w1._layer.weight"
        w2_weight_name = f"model.layers.{layer_id}.{layer_type}.w2._layer.weight"
        w3_weight_name = f"model.layers.{layer_id}.{layer_type}.w3._layer.weight"

        w1_scale_name = f"model.layers.{layer_id}.{layer_type}.w1._layer.matmul.weight_scale"
        w2_scale_name = f"model.layers.{layer_id}.{layer_type}.w2._layer.matmul.weight_scale"
        w3_scale_name = f"model.layers.{layer_id}.{layer_type}.w3._layer.matmul.weight_scale"

        w1_weight_param, w1_scale_param, w3_weight_param, w3_scale_param, w2_weight_param = \
            self.get_smooth_quant_moe_shared_expert_weight(w1_weight_name, w1_scale_name, w3_weight_name, w3_scale_name,
                                                           w2_weight_name, src_hf_dir, hf_weight_map)
        w2_scale_param, _ = self.get_safetensor_from_file(
            w2_scale_name, src_hf_dir, hf_weight_map)

        if ffn_concat:
            concat_weight_name = f"model.layers.{layer_id}.{layer_type}.w_gate_hidden._layer.weight"
            concat_weight_param = ms.Tensor(np.concatenate(
                [w1_weight_param, w3_weight_param], axis=0),
                                            dtype=ms.int8)
            parameter_dict[concat_weight_name] = ms.Parameter(
                concat_weight_param,
                name=concat_weight_name,
                requires_grad=False)

            concat_scale_name = f"model.layers.{layer_id}.{layer_type}.w_gate_hidden._layer.matmul.weight_scale"
            concat_scale_param = ms.Tensor(np.concatenate(
                [w1_scale_param, w3_scale_param], axis=0),
                                           dtype=ms.bfloat16)
            parameter_dict[concat_scale_name] = ms.Parameter(
                concat_scale_param,
                name=concat_scale_name,
                requires_grad=False)

        else:
            # w1 w3
            parameter_dict[w1_weight_name] = ms.Parameter(ms.Tensor(
                w1_weight_param, ms.int8),
                                                          name=w1_weight_name,
                                                          requires_grad=False)
            parameter_dict[w3_weight_name] = ms.Parameter(ms.Tensor(
                w3_weight_param, ms.int8),
                                                          name=w3_weight_name,
                                                          requires_grad=False)

            parameter_dict[w1_scale_name] = ms.Parameter(ms.Tensor(
                w1_scale_param, ms.bfloat16),
                                                         name=w1_scale_name,
                                                         requires_grad=False)
            parameter_dict[w3_scale_name] = ms.Parameter(ms.Tensor(
                w3_scale_param, ms.bfloat16),
                                                         name=w3_scale_name,
                                                         requires_grad=False)

        parameter_dict[w2_weight_name] = ms.Parameter(ms.Tensor(
            w2_weight_param, ms.int8),
                                                      name=w2_weight_name,
                                                      requires_grad=False)
        parameter_dict[w2_scale_name] = ms.Parameter(ms.Tensor(
            w2_scale_param, ms.bfloat16),
                                                     name=w2_scale_name,
                                                     requires_grad=False)

    def smooth_quant_process_ffn_weight(self, src_hf_dir, layer_id,
                                        hf_weight_map, parameter_dict,
                                        layer_type):
        """smooth_quant_process_ffn_weight"""

        ffn_concat = self.config.model.model_config.ffn_concat
        w1_weight_name = f"model.layers.{layer_id}.{layer_type}.w1._layer.weight"
        w1_weight_param, _ = self.get_safetensor_from_file_split_tp_group(
            w1_weight_name, src_hf_dir, hf_weight_map, split_axis=0)
        w1_scale_name = f"model.layers.{layer_id}.{layer_type}.w1._layer.matmul.weight_scale"
        w1_scale_param, _ = self.get_safetensor_from_file_split_tp_group(
            w1_scale_name, src_hf_dir, hf_weight_map, split_axis=0)

        w3_weight_name = f"model.layers.{layer_id}.{layer_type}.w3._layer.weight"

        w3_weight_param, _ = self.get_safetensor_from_file_split_tp_group(
            w3_weight_name, src_hf_dir, hf_weight_map, split_axis=0)
        w3_scale_name = f"model.layers.{layer_id}.{layer_type}.w3._layer.matmul.weight_scale"
        w3_scale_param, _ = self.get_safetensor_from_file_split_tp_group(
            w3_scale_name, src_hf_dir, hf_weight_map, split_axis=0)
        w2_weight_name = f"model.layers.{layer_id}.{layer_type}.w2._layer.weight"
        w2_scale_name = f"model.layers.{layer_id}.{layer_type}.w2._layer.matmul.weight_scale"
        w2_weight_param, _ = self.get_safetensor_from_file_split_tp_group(
            w2_weight_name, src_hf_dir, hf_weight_map, split_axis=1)
        w2_scale_param, _ = self.get_safetensor_from_file(
            w2_scale_name, src_hf_dir, hf_weight_map)

        if ffn_concat:
            concat_weight_name = f"model.layers.{layer_id}.{layer_type}.w_gate_hidden._layer.weight"
            concat_weight_param = ms.Tensor(np.concatenate(
                [w1_weight_param, w3_weight_param], axis=0),
                                            dtype=ms.int8)
            parameter_dict[concat_weight_name] = ms.Parameter(
                concat_weight_param,
                name=concat_weight_name,
                requires_grad=False)

            concat_scale_name = f"model.layers.{layer_id}.{layer_type}.w_gate_hidden._layer.matmul.weight_scale"
            concat_scale_param = ms.Tensor(np.concatenate(
                [w1_scale_param, w3_scale_param], axis=0),
                                           dtype=ms.bfloat16)
            parameter_dict[concat_scale_name] = ms.Parameter(
                concat_scale_param,
                name=concat_scale_name,
                requires_grad=False)
        else:
            # w1 w3
            parameter_dict[w1_weight_name] = ms.Parameter(ms.Tensor(
                w1_weight_param, ms.int8),
                                                          name=w1_weight_name,
                                                          requires_grad=False)
            parameter_dict[w3_weight_name] = ms.Parameter(ms.Tensor(
                w3_weight_param, ms.int8),
                                                          name=w3_weight_name,
                                                          requires_grad=False)

            parameter_dict[w1_scale_name] = ms.Parameter(ms.Tensor(
                w1_scale_param, ms.bfloat16),
                                                         name=w1_scale_name,
                                                         requires_grad=False)
            parameter_dict[w3_scale_name] = ms.Parameter(ms.Tensor(
                w3_scale_param, ms.bfloat16),
                                                         name=w3_scale_name,
                                                         requires_grad=False)

        parameter_dict[w2_weight_name] = ms.Parameter(ms.Tensor(
            w2_weight_param, ms.int8),
                                                      name=w2_weight_name,
                                                      requires_grad=False)
        parameter_dict[w2_scale_name] = ms.Parameter(ms.Tensor(
            w2_scale_param, ms.bfloat16),
                                                     name=w2_scale_name,
                                                     requires_grad=False)

    def smooth_quant_process_qkv_weight(self, src_hf_dir, layer_id,
                                        hf_weight_map, parameter_dict):
        '''smooth_quant_process_qkv_weight'''
        qkv_concat = self.config.model.model_config.qkv_concat
        # q2l_proj
        q2l_weight_name = f"model.layers.{layer_id}.attention.q2l_proj._layer.weight"
        q2l_weight_param, _ = self.get_safetensor_from_file(
            q2l_weight_name, src_hf_dir, hf_weight_map)
        q2l_bias_name = f"model.layers.{layer_id}.attention.q2l_proj._layer.matmul.quant_bias"
        q2l_bias_param, _ = self.get_safetensor_from_file(
            q2l_bias_name, src_hf_dir, hf_weight_map)
        q2l_scale_name = f"model.layers.{layer_id}.attention.q2l_proj._layer.matmul.dequant_scale"
        q2l_scale_param, _ = self.get_safetensor_from_file(
            q2l_scale_name, src_hf_dir, hf_weight_map)

        q2l_quant_zp = f"model.layers.{layer_id}.attention.q2l_proj.quant_op.input_zp"
        q2l_quant_scale = f"model.layers.{layer_id}.attention.q2l_proj.quant_op.input_scale"
        q2l_quant_zp_param, _ = self.get_safetensor_from_file(
            q2l_quant_zp, src_hf_dir, hf_weight_map)
        q2l_quant_scale_param, _ = self.get_safetensor_from_file(
            q2l_quant_scale, src_hf_dir, hf_weight_map)

        kv2l_weight_name = f"model.layers.{layer_id}.attention.kv2l._layer.weight"
        kv2l_weight_param, _ = self.get_safetensor_from_file(
            kv2l_weight_name, src_hf_dir, hf_weight_map)
        kv2l_bias_name = f"model.layers.{layer_id}.attention.kv2l._layer.matmul.quant_bias"
        kv2l_bias_param, _ = self.get_safetensor_from_file(
            kv2l_bias_name, src_hf_dir, hf_weight_map)
        kv2l_scale_name = f"model.layers.{layer_id}.attention.kv2l._layer.matmul.dequant_scale"
        kv2l_scale_param, _ = self.get_safetensor_from_file(
            kv2l_scale_name, src_hf_dir, hf_weight_map)

        kv2l_quant_zp = f"model.layers.{layer_id}.attention.kv2l.quant_op.input_zp"
        kv2l_quant_scale = f"model.layers.{layer_id}.attention.kv2l.quant_op.input_scale"
        kv2l_quant_zp_param, _ = self.get_safetensor_from_file(
            kv2l_quant_zp, src_hf_dir, hf_weight_map)
        kv2l_quant_scale_param, _ = self.get_safetensor_from_file(
            kv2l_quant_scale, src_hf_dir, hf_weight_map)

        if qkv_concat:
            qkv2l_weight_name = f"model.layers.{layer_id}.attention.qkv2l._layer.weight"
            qkv2l_bias_name = f"model.layers.{layer_id}.attention.qkv2l._layer.matmul.quant_bias"
            qkv2l_scale_name = f"model.layers.{layer_id}.attention.qkv2l._layer.matmul.dequant_scale"
            qkv2l_quant_zp_name = f"model.layers.{layer_id}.attention.qkv2l.quant_op.input_zp"
            qkv2l_quant_scale_name = f"model.layers.{layer_id}.attention.qkv2l.quant_op.input_scale"

            qkv2l_weight = np.concatenate(
                (q2l_weight_param, kv2l_weight_param), 0)
            parameter_dict[qkv2l_weight_name] = ms.Parameter(
                ms.Tensor(qkv2l_weight, ms.int8),
                name=qkv2l_weight_name,
                requires_grad=False)
            qkv2l_bias = np.concatenate((q2l_bias_param, kv2l_bias_param), 0)
            parameter_dict[qkv2l_bias_name] = ms.Parameter(
                ms.Tensor(qkv2l_bias, ms.int32),
                name=qkv2l_bias_name,
                requires_grad=False)
            qkv2l_scale = np.concatenate((q2l_scale_param, kv2l_scale_param),
                                         0)
            parameter_dict[qkv2l_scale_name] = ms.Parameter(
                ms.Tensor(qkv2l_scale, ms.float32),
                name=qkv2l_scale_name,
                requires_grad=False)
            parameter_dict[qkv2l_quant_zp_name] = ms.Parameter(
                ms.Tensor(q2l_quant_zp_param, ms.int8),
                name=qkv2l_quant_zp_name,
                requires_grad=False)
            parameter_dict[qkv2l_quant_scale_name] = ms.Parameter(
                ms.Tensor(q2l_quant_scale_param, ms.bfloat16),
                name=qkv2l_quant_scale_name,
                requires_grad=False)
        else:
            parameter_dict[q2l_weight_name] = ms.Parameter(
                ms.Tensor(q2l_weight_param, ms.int8),
                name=q2l_weight_name,
                requires_grad=False)
            parameter_dict[kv2l_weight_name] = ms.Parameter(
                ms.Tensor(kv2l_weight_param, ms.int8),
                name=kv2l_weight_name,
                requires_grad=False)
            parameter_dict[q2l_bias_name] = ms.Parameter(ms.Tensor(
                q2l_bias_param, ms.int32),
                                                         name=q2l_bias_name,
                                                         requires_grad=False)
            parameter_dict[kv2l_bias_name] = ms.Parameter(ms.Tensor(
                kv2l_bias_param, ms.int32),
                                                          name=kv2l_bias_name,
                                                          requires_grad=False)
            parameter_dict[q2l_scale_name] = ms.Parameter(ms.Tensor(
                q2l_scale_param, ms.float32),
                                                          name=q2l_scale_name,
                                                          requires_grad=False)
            parameter_dict[kv2l_scale_name] = ms.Parameter(
                ms.Tensor(kv2l_scale_param, ms.float32),
                name=kv2l_scale_name,
                requires_grad=False)
            parameter_dict[q2l_quant_zp] = ms.Parameter(ms.Tensor(
                q2l_quant_zp_param, ms.int8),
                                                        name=q2l_quant_zp,
                                                        requires_grad=False)
            parameter_dict[kv2l_quant_zp] = ms.Parameter(ms.Tensor(
                kv2l_quant_zp_param, ms.int8),
                                                         name=kv2l_quant_zp,
                                                         requires_grad=False)
            parameter_dict[q2l_quant_scale] = ms.Parameter(
                ms.Tensor(q2l_quant_scale_param, ms.bfloat16),
                name=q2l_quant_scale,
                requires_grad=False)
            parameter_dict[kv2l_quant_scale] = ms.Parameter(
                ms.Tensor(kv2l_quant_scale_param, ms.bfloat16),
                name=kv2l_quant_scale,
                requires_grad=False)

    def infer_smooth_quant_row_linear_split(self, param_name, src_hf_dir,
                                            hf_weight_map):
        '''infer_smooth_quant_row_linear_split'''
        if param_name.endswith(".weight"):
            value, _ = self.get_safetensor_from_file_split_tp_group(
                param_name, src_hf_dir, hf_weight_map, split_axis=1)
        elif "quant_op" in param_name:
            value, _ = self.get_safetensor_from_file_split_tp_group(
                param_name, src_hf_dir, hf_weight_map, split_axis=0)
        else:
            value, _ = self.get_safetensor_from_file(param_name, src_hf_dir,
                                                     hf_weight_map)
        quant_bias_set_zero = [
            "wo._layer.matmul.quant_bias", "w2._layer.matmul.quant_bias"
        ]
        if any([name in param_name for name in quant_bias_set_zero]) and \
                get_tensor_model_parallel_rank() != 0:
            value.fill(0)

        return value

    def infer_smooth_quant_get_value(self, param_name, src_hf_dir,
                                     hf_weight_map, no_need_split_layer):
        '''infer_smooth_quant_get_value'''

        if any([name in param_name for name in no_need_split_layer]):
            value, _ = self.get_safetensor_from_file(param_name, src_hf_dir,
                                                     hf_weight_map)
        elif any([name in param_name for name in [".l2q_proj."]]):
            if param_name.endswith(".weight") or "matmul" in param_name:
                value, _ = self.get_safetensor_from_file_split_tp_group(
                    param_name, src_hf_dir, hf_weight_map, split_axis=0)
            else:
                value, _ = self.get_safetensor_from_file(
                    param_name, src_hf_dir, hf_weight_map)
        elif any([name in param_name for name in [".wo."]]):
            value = self.infer_smooth_quant_row_linear_split(
                param_name, src_hf_dir, hf_weight_map)
        elif any(
            [name in param_name for name in ["lkv2kv_k_nope", "lkv2kv_v"]]):
            value, _ = self.get_safetensor_from_file_split_tp_group(
                param_name, src_hf_dir, hf_weight_map, split_axis=0)
        elif "lm_head" in param_name:
            if not self.config.parallel_config.vocab_emb_dp:
                value, _ = self.get_safetensor_from_file_split_tp_group(
                    param_name, src_hf_dir, hf_weight_map, split_axis=0)
            else:
                value, _ = self.get_safetensor_from_file(
                    param_name, src_hf_dir, hf_weight_map)
        else:
            raise ValueError(
                f"not found layer {param_name}, please check safetensors file."
            )
        return value

    def infer_smooth_quant_net_ms_convert_layer_weight(self, src_hf_dir,
                                                       num_layers,
                                                       hf_weight_map):
        '''infer_smooth_quant_net_ms_convert_layer_weight'''
        parameter_dict = {}  # type: ignore[var-annotated]

        no_need_split_layer = [
            "tok_embeddings", "norm", "routed_experts.router.dense",
            "routed_experts.router.e_score_correction_bias", "topk_bias"
        ]
        for layer_id in tqdm(range(num_layers), desc="qkv/ffn params load"):
            if layer_id >= 3:
                self.smooth_quant_process_route_ffn_weight(
                    src_hf_dir, layer_id, hf_weight_map, parameter_dict,
                    "feed_forward.routed_experts.ffn")
                self.smooth_quant_process_shared_ffn_weight(
                    src_hf_dir, layer_id, hf_weight_map, parameter_dict,
                    "feed_forward.shared_experts")

            else:
                self.smooth_quant_process_ffn_weight(src_hf_dir, layer_id,
                                                     hf_weight_map,
                                                     parameter_dict,
                                                     "feed_forward")
            self.smooth_quant_process_qkv_weight(src_hf_dir, layer_id,
                                                 hf_weight_map, parameter_dict)

        skip_layer = [
            "feed_forward.routed_experts.ffn", "feed_forward.shared_experts",
            "feed_forward.w", "attention.kv2l", "attention.q"
        ]

        for param_name, _ in tqdm(hf_weight_map.items(),
                                  desc="remaining params load"):
            if "model.layers" in param_name and int(
                    param_name.split('.')[2]) >= num_layers:
                continue

            if any([name in param_name for name in skip_layer]):
                continue

            value = self.infer_smooth_quant_get_value(param_name, src_hf_dir,
                                                      hf_weight_map,
                                                      no_need_split_layer)
            dst_dtype = convert_np_to_ms_dtype(value)

            parameter_dict[param_name] = ms.Parameter(ms.Tensor(
                value, dtype=dst_dtype),
                                                      name=param_name,
                                                      requires_grad=False)

        param_not_load, ckpt_not_load = ms.load_param_into_net(
            self.network, parameter_dict)
        logger.info("smoothquant param_not_load: %s", param_not_load)
        logger.info("smoothquant ckpt_not_load: %s", ckpt_not_load)

    def infer_gptq_quant_net_ms_convert_layer_weight(self, src_hf_dir,
                                                     num_layers,
                                                     hf_weight_map):
        """infer_gptq_quant_net_ms_convert_layer_weight"""
        parameter_dict = {}

        no_need_split_layer = [
            "tok_embeddings", "norm", "q2l_proj", "kv2l",
            "routed_experts.router.dense",
            "routed_experts.router.e_score_correction_bias", "topk_bias"
        ]

        for param_name, _ in tqdm(hf_weight_map.items(),
                                  desc="split safetensors"):
            if "model.layers" in param_name and int(
                    param_name.split('.')[2]) >= num_layers:
                continue

            if any([name in param_name for name in no_need_split_layer]):
                value, is_int4 = self.get_safetensor_from_file(
                    param_name, src_hf_dir, hf_weight_map)
            elif any([
                    name in param_name for name in [
                        ".l2q_proj.", ".feed_forward.w_gate_hidden.",
                        "shared_experts.w_gate_hidden"
                    ]
            ]):
                value, is_int4 = self.get_safetensor_from_file_split_tp_group(
                    param_name, src_hf_dir, hf_weight_map, split_axis=1)
            elif any([name in param_name for name in [".wo."]]):
                value, is_int4 = self.get_safetensor_from_file_split_tp_group(
                    param_name, src_hf_dir, hf_weight_map, split_axis=0)
            elif any([
                    name in param_name
                    for name in [".feed_forward.w2.", "shared_experts.w2"]
            ]):
                value = self.infer_smooth_quant_row_linear_split(
                    param_name, src_hf_dir, hf_weight_map)
                is_int4 = False
            elif ".routed_experts.ffn.w_gate_hidden." in param_name:
                value, is_int4 = self.get_safetensor_from_file(
                    param_name, src_hf_dir, hf_weight_map)
                value_list = []
                for experts_id in range(value.shape[0]):
                    value_list.append(
                        self.split_weight_by_rank(value[experts_id, :, :],
                                                  split_axis=1))
                value = np.stack(value_list, axis=0)
            elif ".routed_experts.ffn.w2" in param_name:
                value, is_int4 = self.get_safetensor_from_file(
                    param_name, src_hf_dir, hf_weight_map)
                value_list = []
                for experts_id in range(value.shape[0]):
                    value_list.append(
                        self.split_weight_by_rank(value[experts_id, :, :],
                                                  split_axis=0))
                value = np.stack(value_list, axis=0)
            elif any(
                [name in param_name
                 for name in ["lkv2kv_k_nope", "lkv2kv_v"]]):
                value, is_int4 = self.get_safetensor_from_file_split_tp_group(
                    param_name, src_hf_dir, hf_weight_map, split_axis=0)
            elif "lm_head" in param_name:
                if not self.config.parallel_config.vocab_emb_dp:
                    value, is_int4 = self.get_safetensor_from_file_split_tp_group(
                        param_name, src_hf_dir, hf_weight_map, split_axis=0)
                else:
                    value, is_int4 = self.get_safetensor_from_file(
                        param_name, src_hf_dir, hf_weight_map)
            else:
                raise ValueError(
                    f"not found layer {param_name}, please check safetensors file."
                )

            dst_dtype = convert_np_to_ms_dtype(value)
            if is_int4:
                parameter_dict[param_name] = ms.Parameter(ms.Tensor(
                    value, dtype=dtype.qint4x2),
                                                          name=param_name,
                                                          requires_grad=False)
            else:
                parameter_dict[param_name] = ms.Parameter(ms.Tensor(
                    value, dtype=dst_dtype),
                                                          name=param_name,
                                                          requires_grad=False)
            _, _ = ms.load_param_into_net(self.network, parameter_dict)

    def load_safetensors_shard(self, src_hf_dir, is_mtp_model=False):
        """deepseek load safetensors and shard """
        rank_id = get_rank()
        param_json_path = ""

        for file in os.listdir(src_hf_dir):
            if file.endswith('index.json'):
                # mtp model do not support quantization, needs to load bf16 weight.
                if ('quant' in file and self.is_quant) or \
                        ('quant' not in file and (not self.is_quant or is_mtp_model)):
                    param_json_path = os.path.join(src_hf_dir, file)
                    with open(param_json_path) as fp:
                        hf_weight_map = json.load(fp)['weight_map']
                    break
            elif file.endswith('_name_map.json'):
                param_json_path = os.path.join(src_hf_dir, file)
                with open(param_json_path) as fp:
                    hf_weight_map = json.load(fp)
                    if hf_weight_map.get('weight_map'):
                        hf_weight_map = hf_weight_map['weight_map']
                break

        if not param_json_path:
            raise ValueError(f"Not found param_json_path in {src_hf_dir}")

        quantization_config = self.config.model.model_config.quantization_config
        quant_method = quantization_config.quant_method if quantization_config else None
        support_quant_method = ["gptq-pergroup", "smoothquant", "osl"]
        if not quant_method or (quant_method not in support_quant_method) and \
                not is_mtp_model:
            self.infer_convert_outer_weight(src_hf_dir, hf_weight_map)

        if quant_method and quant_method == "gptq-pergroup":
            self.infer_gptq_quant_net_ms_convert_layer_weight(
                src_hf_dir, self.num_layers, hf_weight_map)
            return
        if quant_method and quant_method == "smoothquant":
            self.infer_smooth_quant_net_ms_convert_layer_weight(
                src_hf_dir, self.num_layers, hf_weight_map)
            return
        if quant_method and quant_method == "osl":
            self.infer_smooth_quant_net_ms_convert_layer_weight(
                src_hf_dir, self.num_layers, hf_weight_map)
            return

        enable_tqdm = rank_id == 0
        mtp_layers = self.config.model.model_config.num_nextn_predict_layers
        start_layer = 0 if not is_mtp_model else self.num_layers
        end_layer = self.num_layers if not is_mtp_model else self.num_layers + mtp_layers
        for layer_id in tqdm(range(start_layer, end_layer),
                             desc="Weight loading",
                             disable=not enable_tqdm):
            if self.is_quant:
                self.infer_quant_net_convert_layer_weight(
                    src_hf_dir, layer_id, hf_weight_map)
            else:
                self.infer_convert_layer_weight(src_hf_dir, layer_id,
                                                hf_weight_map)

        param_not_load, ckpt_not_load = ms.load_param_into_net(
            self.network, self.parameter_dict)
        logger.info("param_not_load: %s, ckpt_not_load: %s",
                    str(param_not_load), str(ckpt_not_load))
        del self.parameter_dict
        gc.collect()
