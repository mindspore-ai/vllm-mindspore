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
import os
import time
import json
import numpy as np

import mindspore as ms
from vllm_mindspore.model_executor.models.mf_models.model_parallelism import BaseModelParallelism


class DeepseekInferParallelism(BaseModelParallelism):
    r"""
    Provide DeepseekV3/R1 Quant Model infer parameter convert and parallelism.
    Args:
        config (DeepseekV3/R1Config): The config of DeepseekV3/R1 model.
        network (InferenceDeepseekV3ForCausalLM): The network of DeepseekV3/R1.

    """

    def __init__(self, config, network, is_quant):
        super().__init__(config, network, is_quant)

    def quant_convert_weight_name(self, weight_name: str):
        """replace quant net weight name"""
        weight_name = weight_name.replace('embed_tokens.weight', 'tok_embeddings.embedding_weight')

        weight_name = weight_name.replace('.self_attn.q_a_proj.weight', '.attention.q2l_proj._layer.weight')
        weight_name = weight_name.replace('.self_attn.q_a_proj.input_scale', '.attention.q2l_proj.quant_op.input_scale')
        weight_name = weight_name.replace('.self_attn.q_a_proj.input_offset', '.attention.q2l_proj.quant_op.input_zp')
        weight_name = weight_name.replace('.self_attn.q_a_proj.quant_bias',
                                          '.attention.q2l_proj._layer.matmul.quant_bias')
        weight_name = weight_name.replace('.self_attn.q_a_proj.deq_scale',
                                          '.attention.q2l_proj._layer.matmul.dequant_scale')

        weight_name = weight_name.replace('.self_attn.q_a_layernorm.weight', '.attention.lq_norm.weight')
        weight_name = weight_name.replace('.self_attn.kv_a_layernorm.weight', '.attention.lkv_norm.weight')
        weight_name = weight_name.replace('.self_attn.kv_b_proj.', '.attention.lkv2kv.')

        weight_name = weight_name.replace('.self_attn.q_b_proj.weight', '.attention.l2q_proj._layer.weight')
        weight_name = weight_name.replace('.self_attn.q_b_proj.input_scale', '.attention.l2q_proj.quant_op.input_scale')
        weight_name = weight_name.replace('.self_attn.q_b_proj.input_offset', '.attention.l2q_proj.quant_op.input_zp')
        weight_name = weight_name.replace('.self_attn.q_b_proj.quant_bias',
                                          '.attention.l2q_proj._layer.matmul.quant_bias')
        weight_name = weight_name.replace('.self_attn.q_b_proj.deq_scale',
                                          '.attention.l2q_proj._layer.matmul.dequant_scale')

        weight_name = weight_name.replace('.self_attn.kv_a_proj_with_mqa.weight', '.attention.kv2l._layer.weight')
        weight_name = weight_name.replace('.self_attn.kv_a_proj_with_mqa.input_scale',
                                          '.attention.kv2l.quant_op.input_scale')
        weight_name = weight_name.replace('.self_attn.kv_a_proj_with_mqa.input_offset',
                                          '.attention.kv2l.quant_op.input_zp')
        weight_name = weight_name.replace('.self_attn.kv_a_proj_with_mqa.quant_bias',
                                          '.attention.kv2l._layer.matmul.quant_bias')
        weight_name = weight_name.replace('.self_attn.kv_a_proj_with_mqa.deq_scale',
                                          '.attention.kv2l._layer.matmul.dequant_scale')

        weight_name = weight_name.replace('.self_attn.o_proj.weight', '.attention.wo._layer.weight')
        weight_name = weight_name.replace('.self_attn.o_proj.input_scale', '.attention.wo.quant_op.input_scale')
        weight_name = weight_name.replace('.self_attn.o_proj.input_offset', '.attention.wo.quant_op.input_zp')
        weight_name = weight_name.replace('.self_attn.o_proj.quant_bias', '.attention.wo._layer.matmul.quant_bias')
        weight_name = weight_name.replace('.self_attn.o_proj.deq_scale', '.attention.wo._layer.matmul.dequant_scale')

        weight_name = weight_name.replace('.self_attn.q_a_layernorm.bias', '.attention.l2q_proj.quant_op.beta')
        weight_name = weight_name.replace('.input_layernorm.bias', '.attention.q2l_proj.quant_op.beta')

        # mlp is pertoken quant
        weight_name = weight_name.replace('.weight_scale', '.matmul.weight_scale')
        weight_name = weight_name.replace('.weight_offset', '.matmul.weight_offset')

        weight_name = weight_name.replace('mlp.gate_proj.', 'feed_forward.w1._layer.')
        weight_name = weight_name.replace('mlp.down_proj.', 'feed_forward.w2._layer.')
        weight_name = weight_name.replace('mlp.up_proj.', 'feed_forward.w3._layer.')
        weight_name = weight_name.replace('mlp.experts.', 'feed_forward.routed_experts.ffn.')
        weight_name = weight_name.replace('mlp.shared_experts.gate_proj.', 'feed_forward.shared_experts.w1._layer.')
        weight_name = weight_name.replace('mlp.shared_experts.down_proj.', 'feed_forward.shared_experts.w2._layer.')
        weight_name = weight_name.replace('mlp.shared_experts.up_proj.', 'feed_forward.shared_experts.w3._layer.')
        weight_name = weight_name.replace('mlp.gate.weight', 'feed_forward.routed_experts.router.dense.weight')
        weight_name = weight_name.replace('mlp.gate.e_score_correction_bias',
                                          'feed_forward.routed_experts.router.e_score_correction_bias')
        weight_name = weight_name.replace('.input_layernorm.weight', '.attention_norm.weight')
        weight_name = weight_name.replace('.post_attention_layernorm.', '.ffn_norm.')
        weight_name = weight_name.replace('model.norm.weight', 'model.norm_out.weight')
        return weight_name

    def infer_trans_rope_weight(self, weight, qk_rope_head_dim):
        """process rope router weight"""
        w1 = weight[..., -qk_rope_head_dim::2, :]
        w2 = weight[..., -qk_rope_head_dim + 1::2, :]
        weight[..., -qk_rope_head_dim:, :] = np.concatenate([w1, w2], axis=-2)
        return weight

    def infer_quant_process_moe_routed_expert_ffn_weight(self, src_hf_dir, layer_id, hf_weight_map):
        """process moe router expert weight"""
        ffn_concat = self.config.model.model_config.ffn_concat
        num_router_experts = self.config.moe_config.expert_num

        parameter_dict = {}
        # router expert dense
        router_dense_hf_name = f"model.layers.{layer_id}.mlp.gate.weight"
        router_dense_ms_name = self.quant_convert_weight_name(router_dense_hf_name)
        router_dense_ms_param = self.get_safetensor_from_file(router_dense_hf_name, src_hf_dir, hf_weight_map)
        parameter_dict[router_dense_ms_name] = ms.Parameter(ms.Tensor(router_dense_ms_param, ms.bfloat16),
                                                            name=router_dense_ms_name, requires_grad=False)

        # e_score_correction_bias
        e_score_correction_bias_hf_name = f"model.layers.{layer_id}.mlp.gate.e_score_correction_bias"
        e_score_correction_bias_ms_name = self.quant_convert_weight_name(e_score_correction_bias_hf_name)
        e_score_correction_bias_ms_param = self.get_safetensor_from_file(e_score_correction_bias_hf_name, src_hf_dir,
                                                                         hf_weight_map)
        parameter_dict[e_score_correction_bias_ms_name] = ms.Parameter(
            ms.Tensor(e_score_correction_bias_ms_param, ms.bfloat16),
            name=e_score_correction_bias_ms_name, requires_grad=False)

        w1_list = []
        w2_list = []
        w3_list = []

        w1_scale_list = []
        w2_scale_list = []
        w3_scale_list = []

        w1_ms_name = f"model.layers.{layer_id}.feed_forward.routed_experts.ffn.w1._layer.weight"
        w2_ms_name = f"model.layers.{layer_id}.feed_forward.routed_experts.ffn.w2._layer.weight"
        w3_ms_name = f"model.layers.{layer_id}.feed_forward.routed_experts.ffn.w3._layer.weight"

        w1_scale_ms_name = f"model.layers.{layer_id}.feed_forward.routed_experts.ffn.w1._layer.matmul.weight_scale"
        w2_scale_ms_name = f"model.layers.{layer_id}.feed_forward.routed_experts.ffn.w2._layer.matmul.weight_scale"
        w3_scale_ms_name = f"model.layers.{layer_id}.feed_forward.routed_experts.ffn.w3._layer.matmul.weight_scale"

        for index in range(0, num_router_experts):
            w1_hf_name = f"model.layers.{layer_id}.mlp.experts.{index}.gate_proj.weight"
            w1_ms_param = self.get_safetensor_from_file(w1_hf_name, src_hf_dir, hf_weight_map,
                                                        is_split_param=True, split_axis=0)

            w2_hf_name = f"model.layers.{layer_id}.mlp.experts.{index}.down_proj.weight"
            w2_ms_param = self.get_safetensor_from_file(w2_hf_name, src_hf_dir, hf_weight_map,
                                                        is_split_param=True, split_axis=1)

            w3_hf_name = f"model.layers.{layer_id}.mlp.experts.{index}.up_proj.weight"
            w3_ms_param = self.get_safetensor_from_file(w3_hf_name, src_hf_dir, hf_weight_map,
                                                        is_split_param=True, split_axis=0)

            w1_list.append(w1_ms_param)
            w2_list.append(w2_ms_param)
            w3_list.append(w3_ms_param)

            w1_scale_hf_name = f"model.layers.{layer_id}.mlp.experts.{index}.gate_proj.weight_scale"
            w1_scale_ms_param = self.get_safetensor_from_file(w1_scale_hf_name, src_hf_dir, hf_weight_map,
                                                              is_split_param=True, split_axis=0)

            w2_scale_hf_name = f"model.layers.{layer_id}.mlp.experts.{index}.down_proj.weight_scale"
            w2_scale_ms_param = self.get_safetensor_from_file(w2_scale_hf_name, src_hf_dir, hf_weight_map)
            # is_split_param=True, split_axis=0)

            w3_scale_hf_name = f"model.layers.{layer_id}.mlp.experts.{index}.up_proj.weight_scale"
            w3_scale_ms_param = self.get_safetensor_from_file(w3_scale_hf_name, src_hf_dir, hf_weight_map,
                                                              is_split_param=True, split_axis=0)

            w1_scale_ms_param = w1_scale_ms_param.squeeze(axis=-1)
            w2_scale_ms_param = w2_scale_ms_param.squeeze(axis=-1)
            w3_scale_ms_param = w3_scale_ms_param.squeeze(axis=-1)
            w1_scale_list.append(w1_scale_ms_param)
            w2_scale_list.append(w2_scale_ms_param)
            w3_scale_list.append(w3_scale_ms_param)

        w1_ms_stack_param = np.stack(w1_list, axis=0).transpose(0, 2, 1)
        w2_ms_stack_param = np.stack(w2_list, axis=0).transpose(0, 2, 1)
        w3_ms_stack_param = np.stack(w3_list, axis=0).transpose(0, 2, 1)

        w1_scale_ms_stack_param = np.stack(w1_scale_list, axis=0)
        w2_scale_ms_stack_param = np.stack(w2_scale_list, axis=0)
        w3_scale_ms_stack_param = np.stack(w3_scale_list, axis=0)

        if ffn_concat:
            # w_gate_hidden
            w_gate_hidden_name = f"model.layers.{layer_id}.feed_forward.routed_experts.ffn.w_gate_hidden._layer.weight"
            w_gate_hidden_param = ms.Tensor(np.concatenate([w1_ms_stack_param, w3_ms_stack_param], axis=2),
                                            dtype=ms.int8)
            parameter_dict[w_gate_hidden_name] = ms.Parameter(w_gate_hidden_param, name=w_gate_hidden_name,
                                                              requires_grad=False)
            # w_scale_gate_hidden
            w_scale_gate_hidden_name = f"model.layers.{layer_id}.feed_forward.routed_experts.ffn.w_gate_hidden._layer.matmul.weight_scale"
            w_scale_gate_hidden_param = ms.Tensor(
                np.concatenate([w1_scale_ms_stack_param, w3_scale_ms_stack_param], axis=1), dtype=ms.bfloat16)
            parameter_dict[w_scale_gate_hidden_name] = ms.Parameter(w_scale_gate_hidden_param,
                                                                    name=w_scale_gate_hidden_name,
                                                                    requires_grad=False)
        else:
            # w1 w3
            parameter_dict[w1_ms_name] = ms.Parameter(ms.Tensor(w1_ms_stack_param, ms.int8), name=w1_ms_name,
                                                      requires_grad=False)
            parameter_dict[w3_ms_name] = ms.Parameter(ms.Tensor(w3_ms_stack_param, ms.int8), name=w3_ms_name,
                                                      requires_grad=False)

            # w1_scale w3_scale
            parameter_dict[w1_scale_ms_name] = ms.Parameter(ms.Tensor(w1_scale_ms_stack_param, ms.bfloat16),
                                                            name=w1_ms_name,
                                                            requires_grad=False)
            parameter_dict[w3_scale_ms_name] = ms.Parameter(ms.Tensor(w3_scale_ms_stack_param, ms.bfloat16),
                                                            name=w3_ms_name,
                                                            requires_grad=False)

        parameter_dict[w2_ms_name] = ms.Parameter(ms.Tensor(w2_ms_stack_param, ms.int8), name=w2_ms_name,
                                                  requires_grad=False)

        parameter_dict[w2_scale_ms_name] = ms.Parameter(ms.Tensor(w2_scale_ms_stack_param, ms.bfloat16),
                                                        name=w2_scale_ms_name,
                                                        requires_grad=False)
        param_not_load, ckpt_not_load = ms.load_param_into_net(self.network, parameter_dict)

    def infer_quant_process_moe_shared_expert_ffn_weight(self, src_hf_dir, layer_id, hf_weight_map):
        """infer quant process moe shared expert ffn weight"""

        ffn_concat = self.config.model.model_config.ffn_concat
        parameter_dict = {}
        w1_hf_name = f"model.layers.{layer_id}.mlp.shared_experts.gate_proj.weight"
        w1_ms_name = self.quant_convert_weight_name(w1_hf_name)
        w1_ms_param = self.get_safetensor_from_file(w1_hf_name, src_hf_dir, hf_weight_map)

        w1_scale_hf_name = f"model.layers.{layer_id}.mlp.shared_experts.gate_proj.weight_scale"
        w1_scale_ms_name = self.quant_convert_weight_name(w1_scale_hf_name)
        w1_scale_ms_param = self.get_safetensor_from_file(w1_scale_hf_name, src_hf_dir, hf_weight_map)

        w2_hf_name = f"model.layers.{layer_id}.mlp.shared_experts.down_proj.weight"
        w2_ms_name = self.quant_convert_weight_name(w2_hf_name)
        w2_ms_param = self.get_safetensor_from_file(w2_hf_name, src_hf_dir, hf_weight_map)

        w2_scale_hf_name = f"model.layers.{layer_id}.mlp.shared_experts.down_proj.weight_scale"
        w2_scale_ms_name = self.quant_convert_weight_name(w2_scale_hf_name)
        w2_scale_ms_param = self.get_safetensor_from_file(w2_scale_hf_name, src_hf_dir, hf_weight_map)

        w3_hf_name = f"model.layers.{layer_id}.mlp.shared_experts.up_proj.weight"
        w3_ms_name = self.quant_convert_weight_name(w3_hf_name)
        w3_ms_param = self.get_safetensor_from_file(w3_hf_name, src_hf_dir, hf_weight_map)

        w3_scale_hf_name = f"model.layers.{layer_id}.mlp.shared_experts.up_proj.weight_scale"
        w3_scale_ms_name = self.quant_convert_weight_name(w3_scale_hf_name)
        w3_scale_ms_param = self.get_safetensor_from_file(w3_scale_hf_name, src_hf_dir, hf_weight_map)

        w1_scale_ms_param = w1_scale_ms_param.squeeze(axis=-1)
        w2_scale_ms_param = w2_scale_ms_param.squeeze(axis=-1)
        w3_scale_ms_param = w3_scale_ms_param.squeeze(axis=-1)

        if ffn_concat:
            w_gate_hidden_name = f"model.layers.{layer_id}.feed_forward.shared_experts.w_gate_hidden._layer.weight"
            w_gate_hidden_param = ms.Tensor(np.concatenate([w1_ms_param, w3_ms_param], axis=0), dtype=ms.int8)
            parameter_dict[w_gate_hidden_name] = ms.Parameter(w_gate_hidden_param, name=w_gate_hidden_name,
                                                              requires_grad=False)

            w_scale_gate_hidden_name = f"model.layers.{layer_id}.feed_forward.shared_experts.w_gate_hidden._layer.matmul.weight_scale"
            w_scale_gate_hidden_param = ms.Tensor(
                np.concatenate([w1_scale_ms_param, w3_scale_ms_param], axis=0),
                dtype=ms.bfloat16)

            parameter_dict[w_scale_gate_hidden_name] = ms.Parameter(w_scale_gate_hidden_param,
                                                                    name=w_scale_gate_hidden_name,
                                                                    requires_grad=False)

        else:
            parameter_dict[w1_ms_name] = ms.Parameter(ms.Tensor(w1_ms_param, ms.int8),
                                                      name=w1_ms_name,
                                                      requires_grad=False)
            parameter_dict[w3_ms_name] = ms.Parameter(ms.Tensor(w3_ms_param, ms.int8),
                                                      name=w3_ms_name,
                                                      requires_grad=False)

            parameter_dict[w1_scale_ms_name] = ms.Parameter(ms.Tensor(w1_scale_ms_param, ms.bfloat16),
                                                            name=w1_ms_name,
                                                            requires_grad=False)
            parameter_dict[w3_scale_ms_name] = ms.Parameter(ms.Tensor(w3_scale_ms_param, ms.bfloat16),
                                                            name=w3_ms_name,
                                                            requires_grad=False)

        parameter_dict[w2_ms_name] = ms.Parameter(ms.Tensor(w2_ms_param, ms.int8),
                                                  name=w2_ms_name,
                                                  requires_grad=False)

        parameter_dict[w2_scale_ms_name] = ms.Parameter(ms.Tensor(w2_scale_ms_param, ms.bfloat16),
                                                        name=w2_ms_name,
                                                        requires_grad=False)
        param_not_load, ckpt_not_load = ms.load_param_into_net(self.network, parameter_dict)

    def infer_quant_process_dense_ffn_weight(self, src_hf_dir, layer_id, hf_weight_map):
        """infer process dense ffn weight"""

        ffn_concat = self.config.model.model_config.ffn_concat
        parameter_dict = {}
        w1_hf_name = f"model.layers.{layer_id}.mlp.gate_proj.weight"
        w1_ms_name = self.quant_convert_weight_name(w1_hf_name)
        w1_ms_param = self.get_safetensor_from_file(w1_hf_name, src_hf_dir, hf_weight_map,
                                                    is_split_param=True,
                                                    split_axis=0)
        w1_scale_hf_name = f"model.layers.{layer_id}.mlp.gate_proj.weight_scale"
        w1_scale_ms_name = self.quant_convert_weight_name(w1_scale_hf_name)
        w1_scale_ms_param = self.get_safetensor_from_file(w1_scale_hf_name, src_hf_dir, hf_weight_map,
                                                          is_split_param=True,
                                                          split_axis=0)

        w2_hf_name = f"model.layers.{layer_id}.mlp.down_proj.weight"
        w2_ms_name = self.quant_convert_weight_name(w2_hf_name)
        w2_ms_param = self.get_safetensor_from_file(w2_hf_name, src_hf_dir, hf_weight_map,
                                                    is_split_param=True,
                                                    split_axis=1)
        w2_scale_hf_name = f"model.layers.{layer_id}.mlp.down_proj.weight_scale"
        w2_scale_ms_name = self.quant_convert_weight_name(w2_scale_hf_name)
        # shape:[7168,1]
        w2_scale_ms_param = self.get_safetensor_from_file(w2_scale_hf_name, src_hf_dir, hf_weight_map)
        # is_split_param=True,
        # split_axis=0)

        w3_hf_name = f"model.layers.{layer_id}.mlp.up_proj.weight"
        w3_ms_name = self.quant_convert_weight_name(w3_hf_name)
        w3_ms_param = self.get_safetensor_from_file(w3_hf_name, src_hf_dir, hf_weight_map,
                                                    is_split_param=True,
                                                    split_axis=0)
        w3_scale_hf_name = f"model.layers.{layer_id}.mlp.up_proj.weight_scale"
        w3_scale_ms_name = self.quant_convert_weight_name(w3_scale_hf_name)
        w3_scale_ms_param = self.get_safetensor_from_file(w3_scale_hf_name, src_hf_dir, hf_weight_map,
                                                          is_split_param=True,
                                                          split_axis=0)

        w1_scale_ms_param = w1_scale_ms_param.squeeze(axis=-1)
        w2_scale_ms_param = w2_scale_ms_param.squeeze(axis=-1)
        w3_scale_ms_param = w3_scale_ms_param.squeeze(axis=-1)

        if ffn_concat:
            w_gate_hidden_name = f"model.layers.{layer_id}.feed_forward.w_gate_hidden._layer.weight"
            w_gate_hidden_param = ms.Tensor(np.concatenate([w1_ms_param, w3_ms_param], axis=0),
                                            dtype=ms.int8)
            parameter_dict[w_gate_hidden_name] = ms.Parameter(w_gate_hidden_param, name=w_gate_hidden_name,
                                                              requires_grad=False)

            w_scale_gate_hidden_name = f"model.layers.{layer_id}.feed_forward.w_gate_hidden._layer.matmul.weight_scale"
            w_scale_gate_hidden_param = ms.Tensor(
                np.concatenate([w1_scale_ms_param, w3_scale_ms_param], axis=0),
                dtype=ms.bfloat16)
            parameter_dict[w_scale_gate_hidden_name] = ms.Parameter(w_scale_gate_hidden_param,
                                                                    name=w_scale_gate_hidden_name,
                                                                    requires_grad=False)

        else:
            parameter_dict[w1_ms_name] = ms.Parameter(ms.Tensor(w1_ms_param, ms.int8),
                                                      name=w1_ms_name,
                                                      requires_grad=False)
            parameter_dict[w3_ms_name] = ms.Parameter(ms.Tensor(w3_ms_param, ms.int8),
                                                      name=w3_ms_name,
                                                      requires_grad=False)

            parameter_dict[w1_scale_ms_name] = ms.Parameter(ms.Tensor(w1_scale_ms_param, ms.bfloat16),
                                                            name=w1_scale_ms_name,
                                                            requires_grad=False)
            parameter_dict[w3_scale_ms_name] = ms.Parameter(ms.Tensor(w3_scale_ms_param, ms.bfloat16),
                                                            name=w3_scale_ms_name,
                                                            requires_grad=False)

        parameter_dict[w2_ms_name] = ms.Parameter(ms.Tensor(w2_ms_param, ms.int8),
                                                  name=w2_ms_name,
                                                  requires_grad=False)

        parameter_dict[w2_scale_ms_name] = ms.Parameter(ms.Tensor(w2_scale_ms_param, ms.bfloat16),
                                                        name=w2_ms_name,
                                                        requires_grad=False)
        param_not_load, ckpt_not_load = ms.load_param_into_net(self.network, parameter_dict)

    def infer_convert_outer_weight(self, src_hf_dir, hf_weight_map):
        """convert weight not in model"""
        parameter_dict = {}
        embed_tokens_hf_name = "model.embed_tokens.weight"
        embed_tokens_ms_name = self.quant_convert_weight_name(embed_tokens_hf_name)
        np_data = self.get_safetensor_from_file(embed_tokens_hf_name, src_hf_dir, hf_weight_map)
        parameter_dict[embed_tokens_ms_name] = ms.Parameter(ms.Tensor(np_data, ms.bfloat16),
                                                            name=embed_tokens_ms_name,
                                                            requires_grad=False)

        norm_hf_name = "model.norm.weight"
        norm_ms_name = self.quant_convert_weight_name(norm_hf_name)
        np_data = self.get_safetensor_from_file(norm_hf_name, src_hf_dir, hf_weight_map)
        parameter_dict[norm_ms_name] = ms.Parameter(ms.Tensor(np_data, ms.bfloat16), name=norm_ms_name,
                                                    requires_grad=False)

        lm_head_hf_name = "lm_head.weight"
        lm_head_ms_name = self.quant_convert_weight_name(lm_head_hf_name)
        if not self.config.parallel_config.vocab_emb_dp:
            np_data = self.get_safetensor_from_file(lm_head_hf_name, src_hf_dir, hf_weight_map,
                                                    is_split_param=True, split_axis=0)
        else:
            np_data = self.get_safetensor_from_file(lm_head_hf_name, src_hf_dir, hf_weight_map)
        parameter_dict[lm_head_ms_name] = ms.Parameter(ms.Tensor(np_data, ms.bfloat16), name=lm_head_ms_name,
                                                       requires_grad=False)
        param_not_load, ckpt_not_load = ms.load_param_into_net(self.network, parameter_dict)

    def quant_special_attention_weight(self, layer_id, src_hf_dir, hf_weight_map, name, is_trans_rope_weigh=False,
                                       is_split_param=False):
        # q_a_proj->q2l_proj
        # kv_a_proj_with_mqa->kv2l
        # q_a_layernorm->lq_norm
        # o_proj->wo

        # input_scale, input_zp no split
        parameter_dict = {}
        input_scale_hf_name = f"model.layers.{layer_id}.self_attn." + name + ".input_scale"
        input_scale_ms_name = self.quant_convert_weight_name(input_scale_hf_name)
        input_scale_ms_param = self.get_safetensor_from_file(input_scale_hf_name, src_hf_dir, hf_weight_map)
        parameter_dict[input_scale_ms_name] = ms.Parameter(ms.Tensor(input_scale_ms_param, ms.bfloat16),
                                                           name=input_scale_ms_name, requires_grad=False)

        input_zp_hf_name = f"model.layers.{layer_id}.self_attn." + name + ".input_offset"
        input_zp_ms_name = self.quant_convert_weight_name(input_zp_hf_name)
        input_zp_ms_param = self.get_safetensor_from_file(input_zp_hf_name, src_hf_dir, hf_weight_map)
        parameter_dict[input_zp_ms_name] = ms.Parameter(ms.Tensor(input_zp_ms_param, ms.int8),
                                                        name=input_zp_ms_name,
                                                        requires_grad=False)

        if not is_trans_rope_weigh:
            quant_bias_hf_name = f"model.layers.{layer_id}.self_attn." + name + ".quant_bias"
            quant_bias_ms_name = self.quant_convert_weight_name(quant_bias_hf_name)
            quant_bias_ms_param = self.get_safetensor_from_file(quant_bias_hf_name, src_hf_dir, hf_weight_map)

            dequant_scale_hf_name = f"model.layers.{layer_id}.self_attn." + name + ".deq_scale"
            dequant_scale_ms_name = self.quant_convert_weight_name(dequant_scale_hf_name)
            dequant_scale_ms_param = self.get_safetensor_from_file(dequant_scale_hf_name, src_hf_dir, hf_weight_map)
        else:
            kv_lora_rank = self.config.model.model_config.kv_lora_rank
            qk_rope_head_dim = self.config.model.model_config.qk_rope_head_dim
            qk_nope_head_dim = self.config.model.model_config.qk_nope_head_dim

            num_heads = self.config.model.model_config.num_heads
            rope_dim = qk_rope_head_dim + qk_nope_head_dim
            kv_head_dim = kv_lora_rank + qk_rope_head_dim

            quant_bias_hf_name = f"model.layers.{layer_id}.self_attn." + name + ".quant_bias"
            quant_bias_ms_name = self.quant_convert_weight_name(quant_bias_hf_name)
            quant_bias_ms_param = self.get_safetensor_from_file(quant_bias_hf_name, src_hf_dir, hf_weight_map)

            dequant_scale_hf_name = f"model.layers.{layer_id}.self_attn." + name + ".deq_scale"
            dequant_scale_ms_name = self.quant_convert_weight_name(dequant_scale_hf_name)
            dequant_scale_ms_param = self.get_safetensor_from_file(dequant_scale_hf_name, src_hf_dir, hf_weight_map)

            if name == "q_b_proj":
                quant_bias_ms_param = quant_bias_ms_param.reshape(num_heads, rope_dim, -1)
                quant_bias_ms_param = self.infer_trans_rope_weight(quant_bias_ms_param, qk_rope_head_dim)
                quant_bias_ms_param = quant_bias_ms_param.reshape(num_heads * rope_dim, -1).reshape(-1)

                dequant_scale_ms_param = dequant_scale_ms_param.reshape(num_heads, rope_dim, -1)
                dequant_scale_ms_param = self.infer_trans_rope_weight(dequant_scale_ms_param, qk_rope_head_dim)
                dequant_scale_ms_param = dequant_scale_ms_param.reshape(num_heads * rope_dim, -1).reshape(-1)

            elif name == "kv_a_proj_with_mqa":
                quant_bias_ms_param = quant_bias_ms_param.reshape(kv_head_dim, -1)
                quant_bias_ms_param = self.infer_trans_rope_weight(quant_bias_ms_param, qk_rope_head_dim).reshape(-1)

                dequant_scale_ms_param = dequant_scale_ms_param.reshape(kv_head_dim, -1)
                dequant_scale_ms_param = self.infer_trans_rope_weight(dequant_scale_ms_param, qk_rope_head_dim).reshape(
                    -1)

        if is_split_param:
            quant_bias_ms_param = self.split_weight_by_rank(quant_bias_ms_param, split_axis=0)
            dequant_scale_ms_param = self.split_weight_by_rank(dequant_scale_ms_param, split_axis=0)

        parameter_dict[quant_bias_ms_name] = ms.Parameter(ms.Tensor(quant_bias_ms_param, ms.int32),
                                                          name=quant_bias_ms_name, requires_grad=False)
        parameter_dict[dequant_scale_ms_name] = ms.Parameter(ms.Tensor(dequant_scale_ms_param, ms.float32),
                                                             name=dequant_scale_ms_name, requires_grad=False)
        param_not_load, ckpt_not_load = ms.load_param_into_net(self.network, parameter_dict)

    def infer_quant_bias_weight(self, src_hf_dir, layer_id, hf_weight_map):
        # quant_op.beta
        parameter_dict = {}
        q2l_proj_bias_hf_name = f"model.layers.{layer_id}.input_layernorm.bias"
        q2l_proj_bias_ms_name = self.quant_convert_weight_name(q2l_proj_bias_hf_name)
        q2l_proj_bias_ms_param = self.get_safetensor_from_file(q2l_proj_bias_hf_name, src_hf_dir, hf_weight_map)

        kv2l_bias_ms_name = f"model.layers.{layer_id}.attention.kv2l.quant_op.beta"
        kv2l_bias_ms_param = q2l_proj_bias_ms_param.copy()

        l2q_proj_bias_hf_name = f"model.layers.{layer_id}.self_attn.q_a_layernorm.bias"
        l2q_proj_bias_ms_name = self.quant_convert_weight_name(l2q_proj_bias_hf_name)
        l2q_proj_bias_ms_param = self.get_safetensor_from_file(l2q_proj_bias_hf_name, src_hf_dir, hf_weight_map)

        parameter_dict[q2l_proj_bias_ms_name] = ms.Parameter(ms.Tensor(q2l_proj_bias_ms_param, ms.bfloat16),
                                                             name=q2l_proj_bias_ms_name, requires_grad=False)
        parameter_dict[kv2l_bias_ms_name] = ms.Parameter(ms.Tensor(kv2l_bias_ms_param, ms.bfloat16),
                                                         name=kv2l_bias_ms_name,
                                                         requires_grad=False)
        parameter_dict[l2q_proj_bias_ms_name] = ms.Parameter(ms.Tensor(l2q_proj_bias_ms_param, ms.bfloat16),
                                                             name=l2q_proj_bias_ms_name,
                                                             requires_grad=False)
        param_not_load, ckpt_not_load = ms.load_param_into_net(self.network, parameter_dict)

    def infer_quant_process_attention_weight(self, src_hf_dir, layer_id, hf_weight_map):
        """infer quant process attention weight"""
        start_time = time.time()
        parameter_dict = {}
        num_heads = self.config.model.model_config.num_heads
        kv_lora_rank = self.config.model.model_config.kv_lora_rank
        qk_rope_head_dim = self.config.model.model_config.qk_rope_head_dim
        v_head_dim = self.config.model.model_config.v_head_dim
        qk_nope_head_dim = self.config.model.model_config.qk_nope_head_dim

        rope_dim = qk_rope_head_dim + qk_nope_head_dim
        kv_head_dim = kv_lora_rank + qk_rope_head_dim

        # q_a_proj->q2l_proj
        q2l_proj_hf_name = f"model.layers.{layer_id}.self_attn.q_a_proj.weight"
        q2l_proj_ms_name = self.quant_convert_weight_name(q2l_proj_hf_name)
        q2l_proj_ms_param = self.get_safetensor_from_file(q2l_proj_hf_name, src_hf_dir, hf_weight_map)
        parameter_dict[q2l_proj_ms_name] = ms.Parameter(ms.Tensor(q2l_proj_ms_param, ms.int8),
                                                        name=q2l_proj_ms_name,
                                                        requires_grad=False)
        self.quant_special_attention_weight(layer_id, src_hf_dir, hf_weight_map, "q_a_proj")

        # kv_a_proj_with_mqa->kv2l
        kv2l_hf_name = f"model.layers.{layer_id}.self_attn.kv_a_proj_with_mqa.weight"
        kv2l_ms_name = self.quant_convert_weight_name(kv2l_hf_name)
        kv2l_ms_param = self.get_safetensor_from_file(kv2l_hf_name, src_hf_dir, hf_weight_map)
        kv2l_ms_param = kv2l_ms_param.reshape(kv_head_dim, -1)
        kv2l_ms_param = self.infer_trans_rope_weight(kv2l_ms_param, qk_rope_head_dim)
        parameter_dict[kv2l_ms_name] = ms.Parameter(ms.Tensor(kv2l_ms_param, ms.int8), name=kv2l_ms_name,
                                                    requires_grad=False)
        self.quant_special_attention_weight(layer_id, src_hf_dir, hf_weight_map, "kv_a_proj_with_mqa",
                                            is_trans_rope_weigh=True)

        # q_a_layernorm->lq_norm
        lq_norm_hf_name = f"model.layers.{layer_id}.self_attn.q_a_layernorm.weight"
        lq_norm_ms_name = self.quant_convert_weight_name(lq_norm_hf_name)
        lq_norm_ms_param = self.get_safetensor_from_file(lq_norm_hf_name, src_hf_dir, hf_weight_map)
        parameter_dict[lq_norm_ms_name] = ms.Parameter(ms.Tensor(lq_norm_ms_param, ms.bfloat16),
                                                       name=lq_norm_ms_name,
                                                       requires_grad=False)

        # q_b_proj->l2q_proj
        l2q_proj_hf_name = f"model.layers.{layer_id}.self_attn.q_b_proj.weight"
        l2q_proj_ms_name = self.quant_convert_weight_name(l2q_proj_hf_name)
        l2q_proj_ms_param = self.get_safetensor_from_file(l2q_proj_hf_name, src_hf_dir, hf_weight_map)
        l2q_proj_ms_param = l2q_proj_ms_param.reshape(num_heads, rope_dim, -1)
        l2q_proj_ms_param = self.infer_trans_rope_weight(l2q_proj_ms_param, qk_rope_head_dim)
        l2q_proj_ms_param = l2q_proj_ms_param.reshape(num_heads * rope_dim, -1)
        l2q_proj_ms_param = self.split_weight_by_rank(l2q_proj_ms_param, split_axis=0)
        parameter_dict[l2q_proj_ms_name] = ms.Parameter(ms.Tensor(l2q_proj_ms_param, ms.int8),
                                                        name=l2q_proj_ms_name,
                                                        requires_grad=False)
        self.quant_special_attention_weight(layer_id, src_hf_dir, hf_weight_map, "q_b_proj", is_trans_rope_weigh=True,
                                            is_split_param=True)

        # kv_a_layernorm->lkv_norm
        lkv_norm_hf_name = f"model.layers.{layer_id}.self_attn.kv_a_layernorm.weight"
        lkv_norm_ms_name = self.quant_convert_weight_name(lkv_norm_hf_name)
        lkv_norm_ms_param = self.get_safetensor_from_file(lkv_norm_hf_name, src_hf_dir, hf_weight_map)
        parameter_dict[lkv_norm_ms_name] = ms.Parameter(ms.Tensor(lkv_norm_ms_param, ms.bfloat16),
                                                        name=lkv_norm_ms_name,
                                                        requires_grad=False)

        # kv_b_proj->lkv2kv
        lkv2kv_hf_name = f"model.layers.{layer_id}.self_attn.kv_b_proj.weight"
        lkv2kv_ms_name = self.quant_convert_weight_name(lkv2kv_hf_name)
        lkv2kv_ms_param = self.get_safetensor_from_file(lkv2kv_hf_name, src_hf_dir, hf_weight_map)
        lkv2kv_head = qk_nope_head_dim + v_head_dim
        lkv2kv_ms_param = lkv2kv_ms_param.reshape(num_heads, lkv2kv_head, -1)
        value_k_nope, value_v = lkv2kv_ms_param[:, :qk_nope_head_dim, :], lkv2kv_ms_param[:, qk_nope_head_dim:, :]

        # value_k_nope
        value_k_nope = value_k_nope.reshape(-1, value_k_nope.shape[-1])
        value_k_nope = self.split_weight_by_rank(value_k_nope, split_axis=0)
        name_k_nope = lkv2kv_ms_name.replace(".attention.lkv2kv.", ".attention.lkv2kv_k_nope.")
        parameter_dict[name_k_nope] = ms.Parameter(ms.Tensor(value_k_nope, ms.bfloat16), name=name_k_nope,
                                                   requires_grad=False)
        # value_v
        value_v = value_v.reshape(-1, value_v.shape[-1])
        value_v = self.split_weight_by_rank(value_v, split_axis=0)
        name_v = lkv2kv_ms_name.replace(".attention.lkv2kv.", ".attention.lkv2kv_v.")
        parameter_dict[name_v] = ms.Parameter(ms.Tensor(value_v, ms.bfloat16), name=name_v,
                                              requires_grad=False)

        # o_proj->wo
        wo_hf_name = f"model.layers.{layer_id}.self_attn.o_proj.weight"
        wo_ms_name = self.quant_convert_weight_name(wo_hf_name)
        wo_ms_param = self.get_safetensor_from_file(wo_hf_name, src_hf_dir, hf_weight_map)
        wo_ms_param = self.split_weight_by_rank(wo_ms_param, split_axis=1)
        parameter_dict[wo_ms_name] = ms.Parameter(ms.Tensor(wo_ms_param, ms.int8), name=wo_ms_name,
                                                  requires_grad=False)
        self.quant_special_attention_weight(layer_id, src_hf_dir, hf_weight_map, "o_proj")

        param_not_load, ckpt_not_load = ms.load_param_into_net(self.network, parameter_dict)

    def infer_quant_net_convert_layer_weight(self, src_hf_dir, layer_id, hf_weight_map):
        """infer quant net convert layer weight"""
        print(f"..... start convert layer {layer_id} .......", flush=True)

        if layer_id >= 3:
            self.infer_quant_process_moe_routed_expert_ffn_weight(src_hf_dir, layer_id, hf_weight_map)
            self.infer_quant_process_moe_shared_expert_ffn_weight(src_hf_dir, layer_id, hf_weight_map)
        else:
            self.infer_quant_process_dense_ffn_weight(src_hf_dir, layer_id, hf_weight_map)

        self.infer_quant_process_attention_weight(src_hf_dir, layer_id, hf_weight_map)
        self.infer_quant_bias_weight(src_hf_dir, layer_id, hf_weight_map)
        self.infer_process_norm_weight(src_hf_dir, layer_id, hf_weight_map)

        print(f"..... end convert layer {layer_id} .......", flush=True)

    def convert_weight_name(self, weight_name: str):
        """replace weight name"""
        weight_name = weight_name.replace('embed_tokens.weight', 'tok_embeddings.embedding_weight')
        weight_name = weight_name.replace('.self_attn.q_a_proj.', '.attention.q2l_proj.')
        weight_name = weight_name.replace('.self_attn.q_a_layernorm.', '.attention.lq_norm.')
        weight_name = weight_name.replace('.self_attn.q_b_proj.', '.attention.l2q_proj.')
        weight_name = weight_name.replace('.self_attn.kv_a_proj_with_mqa.', '.attention.kv2l.')
        weight_name = weight_name.replace('.self_attn.kv_a_layernorm.', '.attention.lkv_norm.')
        weight_name = weight_name.replace('.self_attn.kv_b_proj.', '.attention.lkv2kv.')
        weight_name = weight_name.replace('.self_attn.o_proj.', '.attention.wo.')
        weight_name = weight_name.replace('mlp.gate_proj.', 'feed_forward.w1.')
        weight_name = weight_name.replace('mlp.down_proj.', 'feed_forward.w2.')
        weight_name = weight_name.replace('mlp.up_proj.', 'feed_forward.w3.')
        weight_name = weight_name.replace('mlp.experts.', 'feed_forward.routed_experts.ffn.')
        weight_name = weight_name.replace('mlp.shared_experts.gate_proj.', 'feed_forward.shared_experts.w1.')
        weight_name = weight_name.replace('mlp.shared_experts.down_proj.', 'feed_forward.shared_experts.w2.')
        weight_name = weight_name.replace('mlp.shared_experts.up_proj.', 'feed_forward.shared_experts.w3.')
        weight_name = weight_name.replace('mlp.gate.weight', 'feed_forward.routed_experts.router.dense.weight')
        weight_name = weight_name.replace('mlp.gate.e_score_correction_bias',
                                          'feed_forward.routed_experts.router.e_score_correction_bias')
        weight_name = weight_name.replace('.input_layernorm.', '.attention_norm.')
        weight_name = weight_name.replace('.post_attention_layernorm.', '.ffn_norm.')
        weight_name = weight_name.replace('model.norm.weight', 'model.norm_out.weight')
        return weight_name

    def infer_process_moe_routed_expert_ffn_weight(self, src_hf_dir, layer_id, hf_weight_map):
        """process moe router expert weight"""
        ffn_concat = self.config.model.model_config.ffn_concat
        num_router_experts = self.config.moe_config.expert_num
        parameter_dict = {}

        # router expert dense
        router_dense_hf_name = f"model.layers.{layer_id}.mlp.gate.weight"
        router_dense_ms_name = self.convert_weight_name(router_dense_hf_name)
        router_dense_ms_param = self.get_safetensor_from_file(router_dense_hf_name, src_hf_dir, hf_weight_map)
        parameter_dict[router_dense_ms_name] = ms.Parameter(ms.Tensor(router_dense_ms_param, ms.bfloat16),
                                                            name=router_dense_ms_name, requires_grad=False)

        # e_score_correction_bias
        e_score_correction_bias_hf_name = f"model.layers.{layer_id}.mlp.gate.e_score_correction_bias"
        e_score_correction_bias_ms_name = self.convert_weight_name(e_score_correction_bias_hf_name)
        e_score_correction_bias_ms_param = self.get_safetensor_from_file(e_score_correction_bias_hf_name, src_hf_dir,
                                                                         hf_weight_map)
        parameter_dict[e_score_correction_bias_ms_name] = ms.Parameter(
            ms.Tensor(e_score_correction_bias_ms_param, ms.bfloat16),
            name=e_score_correction_bias_ms_name, requires_grad=False)

        w1_list = []
        w2_list = []
        w3_list = []

        w1_ms_name = f"model.layers.{layer_id}.feed_forward.routed_experts.ffn.w1.weight"
        w2_ms_name = f"model.layers.{layer_id}.feed_forward.routed_experts.ffn.w2.weight"
        w3_ms_name = f"model.layers.{layer_id}.feed_forward.routed_experts.ffn.w3.weight"
        for index in range(0, num_router_experts):
            w1_hf_name = f"model.layers.{layer_id}.mlp.experts.{index}.gate_proj.weight"
            w1_ms_param = self.get_safetensor_from_file(w1_hf_name, src_hf_dir, hf_weight_map,
                                                        is_split_param=True, split_axis=0)

            w2_hf_name = f"model.layers.{layer_id}.mlp.experts.{index}.down_proj.weight"
            w2_ms_param = self.get_safetensor_from_file(w2_hf_name, src_hf_dir, hf_weight_map,
                                                        is_split_param=True, split_axis=1)

            w3_hf_name = f"model.layers.{layer_id}.mlp.experts.{index}.up_proj.weight"
            w3_ms_param = self.get_safetensor_from_file(w3_hf_name, src_hf_dir, hf_weight_map,
                                                        is_split_param=True, split_axis=0)

            w1_list.append(w1_ms_param)
            w2_list.append(w2_ms_param)
            w3_list.append(w3_ms_param)

        w1_ms_stack_param = np.stack(w1_list, axis=0).transpose(0, 2, 1)
        w2_ms_stack_param = np.stack(w2_list, axis=0).transpose(0, 2, 1)
        w3_ms_stack_param = np.stack(w3_list, axis=0).transpose(0, 2, 1)

        if ffn_concat:
            w_gate_hidden_name = f"model.layers.{layer_id}.feed_forward.routed_experts.ffn.w_gate_hidden.weight"
            w_gate_hidden_param = ms.Tensor(np.concatenate([w1_ms_stack_param, w3_ms_stack_param], axis=2),
                                            dtype=ms.bfloat16)
            parameter_dict[w_gate_hidden_name] = ms.Parameter(w_gate_hidden_param, name=w_gate_hidden_name,
                                                              requires_grad=False)
        else:
            parameter_dict[w1_ms_name] = ms.Parameter(ms.Tensor(w1_ms_stack_param, ms.bfloat16), name=w1_ms_name,
                                                      requires_grad=False)
            parameter_dict[w3_ms_name] = ms.Parameter(ms.Tensor(w3_ms_stack_param, ms.bfloat16), name=w3_ms_name,
                                                      requires_grad=False)

        parameter_dict[w2_ms_name] = ms.Parameter(ms.Tensor(w2_ms_stack_param, ms.bfloat16), name=w2_ms_name,
                                                  requires_grad=False)
        _, ckpt_not_load = ms.load_param_into_net(self.network, parameter_dict)

    def infer_process_moe_shared_expert_ffn_weight(self, src_hf_dir, layer_id, hf_weight_map):
        """infer process moe shared expert ffn weight"""
        ffn_concat = self.config.model.model_config.ffn_concat
        parameter_dict = {}
        w1_hf_name = f"model.layers.{layer_id}.mlp.shared_experts.gate_proj.weight"
        w1_ms_name = self.convert_weight_name(w1_hf_name)
        w1_ms_param = self.get_safetensor_from_file(w1_hf_name, src_hf_dir, hf_weight_map)

        w2_hf_name = f"model.layers.{layer_id}.mlp.shared_experts.down_proj.weight"
        w2_ms_name = self.convert_weight_name(w2_hf_name)
        w2_ms_param = self.get_safetensor_from_file(w2_hf_name, src_hf_dir, hf_weight_map)

        w3_hf_name = f"model.layers.{layer_id}.mlp.shared_experts.up_proj.weight"
        w3_ms_name = self.convert_weight_name(w3_hf_name)
        w3_ms_param = self.get_safetensor_from_file(w3_hf_name, src_hf_dir, hf_weight_map)

        if ffn_concat:
            w_gate_hidden_name = f"model.layers.{layer_id}.feed_forward.shared_experts.w_gate_hidden.weight"
            w_gate_hidden_param = ms.Tensor(np.concatenate([w1_ms_param, w3_ms_param], axis=0), dtype=ms.bfloat16)
            parameter_dict[w_gate_hidden_name] = ms.Parameter(w_gate_hidden_param, name=w_gate_hidden_name,
                                                              requires_grad=False)
        else:
            parameter_dict[w1_ms_name] = ms.Parameter(ms.Tensor(w1_ms_param, ms.bfloat16), name=w1_ms_name,
                                                      requires_grad=False)
            parameter_dict[w3_ms_name] = ms.Parameter(ms.Tensor(w3_ms_param, ms.bfloat16), name=w3_ms_name,
                                                      requires_grad=False)
        parameter_dict[w2_ms_name] = ms.Parameter(ms.Tensor(w2_ms_param, ms.bfloat16), name=w2_ms_name,
                                                  requires_grad=False)
        _, ckpt_not_load = ms.load_param_into_net(self.network, parameter_dict)

    def infer_process_dense_ffn_weight(self, src_hf_dir, layer_id, hf_weight_map):
        """infer process dense ffn weight"""

        ffn_concat = self.config.model.model_config.ffn_concat
        parameter_dict = {}

        w1_hf_name = f"model.layers.{layer_id}.mlp.gate_proj.weight"
        w1_ms_name = self.convert_weight_name(w1_hf_name)
        w1_ms_param = self.get_safetensor_from_file(w1_hf_name, src_hf_dir, hf_weight_map, is_split_param=True,
                                                    split_axis=0)

        w2_hf_name = f"model.layers.{layer_id}.mlp.down_proj.weight"
        w2_ms_name = self.convert_weight_name(w2_hf_name)
        w2_ms_param = self.get_safetensor_from_file(w2_hf_name, src_hf_dir, hf_weight_map, is_split_param=True,
                                                    split_axis=1)

        w3_hf_name = f"model.layers.{layer_id}.mlp.up_proj.weight"
        w3_ms_name = self.convert_weight_name(w3_hf_name)
        w3_ms_param = self.get_safetensor_from_file(w3_hf_name, src_hf_dir, hf_weight_map, is_split_param=True,
                                                    split_axis=0)

        if ffn_concat:
            w_gate_hidden_name = f"model.layers.{layer_id}.feed_forward.w_gate_hidden.weight"
            w_gate_hidden_param = ms.Tensor(np.concatenate([w1_ms_param, w3_ms_param], axis=0), dtype=ms.bfloat16)
            parameter_dict[w_gate_hidden_name] = ms.Parameter(w_gate_hidden_param, name=w_gate_hidden_name,
                                                              requires_grad=False)
        else:
            parameter_dict[w1_ms_name] = ms.Parameter(ms.Tensor(w1_ms_param, ms.bfloat16), name=w1_ms_name,
                                                      requires_grad=False)
            parameter_dict[w3_ms_name] = ms.Parameter(ms.Tensor(w3_ms_param, ms.bfloat16), name=w3_ms_name,
                                                      requires_grad=False)

        parameter_dict[w2_ms_name] = ms.Parameter(ms.Tensor(w2_ms_param, ms.bfloat16), name=w2_ms_name,
                                                  requires_grad=False)
        _, ckpt_not_load = ms.load_param_into_net(self.network, parameter_dict)

    def infer_process_attention_weight(self, src_hf_dir, layer_id, hf_weight_map):
        """infer process attention weight"""
        num_heads = self.config.model.model_config.num_heads
        kv_lora_rank = self.config.model.model_config.kv_lora_rank
        qk_rope_head_dim = self.config.model.model_config.qk_rope_head_dim
        v_head_dim = self.config.model.model_config.v_head_dim
        qk_nope_head_dim = self.config.model.model_config.qk_nope_head_dim

        rope_dim = qk_rope_head_dim + qk_nope_head_dim
        kv_head_dim = kv_lora_rank + qk_rope_head_dim

        parameter_dict = {}
        # q2l_proj
        q2l_proj_hf_name = f"model.layers.{layer_id}.self_attn.q_a_proj.weight"
        q2l_proj_ms_name = self.convert_weight_name(q2l_proj_hf_name)
        q_a_proj_ms_param = self.get_safetensor_from_file(q2l_proj_hf_name, src_hf_dir, hf_weight_map)
        parameter_dict[q2l_proj_ms_name] = ms.Parameter(ms.Tensor(q_a_proj_ms_param, ms.bfloat16),
                                                        name=q2l_proj_ms_name,
                                                        requires_grad=False)

        # kv2l
        kv2l_hf_name = f"model.layers.{layer_id}.self_attn.kv_a_proj_with_mqa.weight"
        kv2l_ms_name = self.convert_weight_name(kv2l_hf_name)
        kv2l_ms_param = self.get_safetensor_from_file(kv2l_hf_name, src_hf_dir, hf_weight_map)
        kv2l_ms_param = kv2l_ms_param.reshape(kv_head_dim, -1)
        kv2l_ms_param = self.infer_trans_rope_weight(kv2l_ms_param, qk_rope_head_dim)
        parameter_dict[kv2l_ms_name] = ms.Parameter(ms.Tensor(kv2l_ms_param, ms.bfloat16), name=kv2l_ms_name,
                                                    requires_grad=False)

        # lq_norm
        lq_norm_hf_name = f"model.layers.{layer_id}.self_attn.q_a_layernorm.weight"
        lq_norm_ms_name = self.convert_weight_name(lq_norm_hf_name)
        lq_norm_ms_param = self.get_safetensor_from_file(lq_norm_hf_name, src_hf_dir, hf_weight_map)
        parameter_dict[lq_norm_ms_name] = ms.Parameter(ms.Tensor(lq_norm_ms_param, ms.bfloat16), name=lq_norm_ms_name,
                                                       requires_grad=False)

        # l2q_proj
        l2q_proj_hf_name = f"model.layers.{layer_id}.self_attn.q_b_proj.weight"
        l2q_proj_ms_name = self.convert_weight_name(l2q_proj_hf_name)
        l2q_proj_ms_param = self.get_safetensor_from_file(l2q_proj_hf_name, src_hf_dir, hf_weight_map)
        l2q_proj_ms_param = l2q_proj_ms_param.reshape(num_heads, rope_dim, -1)
        l2q_proj_ms_param = self.infer_trans_rope_weight(l2q_proj_ms_param, qk_rope_head_dim)
        l2q_proj_ms_param = l2q_proj_ms_param.reshape(num_heads * rope_dim, -1)
        l2q_proj_ms_param = self.split_weight_by_rank(l2q_proj_ms_param, split_axis=0)
        parameter_dict[l2q_proj_ms_name] = ms.Parameter(ms.Tensor(l2q_proj_ms_param, ms.bfloat16),
                                                        name=l2q_proj_ms_name,
                                                        requires_grad=False)

        # lkv_norm
        lkv_norm_hf_name = f"model.layers.{layer_id}.self_attn.kv_a_layernorm.weight"
        lkv_norm_ms_name = self.convert_weight_name(lkv_norm_hf_name)
        lkv_norm_ms_param = self.get_safetensor_from_file(lkv_norm_hf_name, src_hf_dir, hf_weight_map)
        parameter_dict[lkv_norm_ms_name] = ms.Parameter(ms.Tensor(lkv_norm_ms_param, ms.bfloat16),
                                                        name=lkv_norm_ms_name,
                                                        requires_grad=False)

        # lkv2kv
        lkv2kv_hf_name = f"model.layers.{layer_id}.self_attn.kv_b_proj.weight"
        lkv2kv_ms_name = self.convert_weight_name(lkv2kv_hf_name)
        lkv2kv_ms_param = self.get_safetensor_from_file(lkv2kv_hf_name, src_hf_dir, hf_weight_map)
        lkv2kv_head = qk_nope_head_dim + v_head_dim
        lkv2kv_ms_param = lkv2kv_ms_param.reshape(num_heads, lkv2kv_head, -1)
        value_k_nope, value_v = lkv2kv_ms_param[:, :qk_nope_head_dim, :], lkv2kv_ms_param[:, qk_nope_head_dim:, :]

        # value_k_nope
        value_k_nope = value_k_nope.reshape(-1, value_k_nope.shape[-1])
        value_k_nope = self.split_weight_by_rank(value_k_nope, split_axis=0)
        name_k_nope = lkv2kv_ms_name.replace(".attention.lkv2kv.", ".attention.lkv2kv_k_nope.")
        parameter_dict[name_k_nope] = ms.Parameter(ms.Tensor(value_k_nope, ms.bfloat16), name=name_k_nope,
                                                   requires_grad=False)
        # value_v
        value_v = value_v.reshape(-1, value_v.shape[-1])
        value_v = self.split_weight_by_rank(value_v, split_axis=0)
        name_v = lkv2kv_ms_name.replace(".attention.lkv2kv.", ".attention.lkv2kv_v.")
        parameter_dict[name_v] = ms.Parameter(ms.Tensor(value_v, ms.bfloat16), name=name_v,
                                              requires_grad=False)

        # wo
        wo_hf_name = f"model.layers.{layer_id}.self_attn.o_proj.weight"
        wo_ms_name = self.convert_weight_name(wo_hf_name)
        wo_ms_param = self.get_safetensor_from_file(wo_hf_name, src_hf_dir, hf_weight_map)
        wo_ms_param = self.split_weight_by_rank(wo_ms_param, split_axis=1)
        parameter_dict[wo_ms_name] = ms.Parameter(ms.Tensor(wo_ms_param, ms.bfloat16), name=wo_ms_name,
                                                  requires_grad=False)
        _, ckpt_not_load = ms.load_param_into_net(self.network, parameter_dict)

    def infer_process_norm_weight(self, src_hf_dir, layer_id, hf_weight_map):
        """infer process attention weight"""
        parameter_dict = {}
        # attention_norm
        attention_norm_hf_name = f"model.layers.{layer_id}.input_layernorm.weight"
        attention_norm_ms_name = self.convert_weight_name(attention_norm_hf_name)
        attention_norm_ms_param = self.get_safetensor_from_file(attention_norm_hf_name,
                                                                src_hf_dir,
                                                                hf_weight_map)
        parameter_dict[attention_norm_ms_name] = ms.Parameter(ms.Tensor(attention_norm_ms_param, ms.bfloat16),
                                                              name=attention_norm_ms_name,
                                                              requires_grad=False)

        # ffn_norm
        ffn_norm_hf_name = f"model.layers.{layer_id}.post_attention_layernorm.weight"
        ffn_norm_ms_name = self.convert_weight_name(ffn_norm_hf_name)
        ffn_norm_ms_param = self.get_safetensor_from_file(ffn_norm_hf_name, src_hf_dir, hf_weight_map)
        parameter_dict[ffn_norm_ms_name] = ms.Parameter(ms.Tensor(ffn_norm_ms_param, ms.bfloat16),
                                                        name=ffn_norm_ms_name,
                                                        requires_grad=False)

        _, ckpt_not_load = ms.load_param_into_net(self.network, parameter_dict)

    def infer_convert_layer_weight(self, src_hf_dir, layer_id, hf_weight_map):
        """infer convert layer weight"""
        print(f"..... start convert layer {layer_id} .......", flush=True)

        if layer_id >= 3:
            self.infer_process_moe_routed_expert_ffn_weight(src_hf_dir, layer_id, hf_weight_map)
            self.infer_process_moe_shared_expert_ffn_weight(src_hf_dir, layer_id, hf_weight_map)
        else:
            self.infer_process_dense_ffn_weight(src_hf_dir, layer_id, hf_weight_map)

        self.infer_process_attention_weight(src_hf_dir, layer_id, hf_weight_map)
        self.infer_process_norm_weight(src_hf_dir, layer_id, hf_weight_map)

        print(f"..... end convert layer {layer_id} .......", flush=True)

    def infer_convert_and_parallelism(self, src_hf_dir):
        """convert inference model weight """
        param_json_path = ""
        for file in os.listdir(src_hf_dir):
            if file.endswith('index.json'):
                param_json_path = os.path.join(src_hf_dir, file)
                break
        if not param_json_path:
            raise ValueError("param_json_path:{} is error.".format(param_json_path))
        print("param_json_path is {}".format(param_json_path))

        with open(param_json_path, "r") as fp:
            hf_weight_map = json.load(fp)['weight_map']

        self.infer_convert_outer_weight(src_hf_dir, hf_weight_map)
        num_layers = self.config.model.model_config.num_layers
        for layer_id in range(num_layers):
            if self.is_quant:
                self.infer_quant_net_convert_layer_weight(src_hf_dir, layer_id, hf_weight_map)
            else:
                self.infer_convert_layer_weight(src_hf_dir, layer_id, hf_weight_map)
