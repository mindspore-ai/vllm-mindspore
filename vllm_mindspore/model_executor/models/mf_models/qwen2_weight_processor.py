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
import json
import gc
import numpy as np
from tqdm import tqdm
from safetensors import safe_open
import mindspore as ms
from mindspore.communication.management import get_rank

from vllm_mindspore.utils import convert_np_to_ms_dtype
from vllm_mindspore.model_executor.models.mf_models.weight_processor import BaseWeightProcessor


class Qwen2WeightProcessor(BaseWeightProcessor):
    r"""
    Provide Qwen2 Model weight load and shards.
    Args:
        config (Qwen2Config): The config of Qwen2 model.
        network (InferenceQwen2ForCausalLM): The network of Qwen2.

    """

    def __init__(self, config, network, is_quant):
        super().__init__(config, network, is_quant)
        self.num_heads = config.model.model_config.num_heads
        self.kv_heads = config.model.model_config.n_kv_heads
        self.hidden_size = config.model.model_config.hidden_size

    def qkv_concat_hf2mg(self, qkv_weights: np.ndarray, num_heads, n_kv_heads, hidden_size):
        """
        convert qkv_concat weight with huggingface format to megatron format.
        """
        w, h = qkv_weights.shape
        n_rep = num_heads // n_kv_heads
        q_channel = hidden_size // self.tp_group_size
        kv_channel = (hidden_size // n_rep) // self.tp_group_size
        q_weight = qkv_weights[: q_channel, :]
        k_weight = qkv_weights[q_channel: q_channel + kv_channel, :]
        v_weight = qkv_weights[q_channel + kv_channel: q_channel + 2 * kv_channel, :]
        q_w_reshape = q_weight.reshape(n_kv_heads // self.tp_group_size, hidden_size // n_kv_heads, -1)
        k_w_reshape = k_weight.reshape(n_kv_heads // self.tp_group_size, hidden_size // num_heads, -1)
        v_w_reshape = v_weight.reshape(n_kv_heads // self.tp_group_size, hidden_size // num_heads, -1)
        cat_qkv_weight = np.concatenate((q_w_reshape, k_w_reshape, v_w_reshape), axis=1)
        out_qkv_weight = cat_qkv_weight.reshape(w, h)
        return out_qkv_weight

    def qkv_bias_concat_hf2mg(self, qkv_bias: np.ndarray, num_heads, n_kv_heads, hidden_size):
        """
        convert qkv_concat bias with huggingface format to megatron format.
        """
        w = qkv_bias.shape[0]
        n_rep = num_heads // n_kv_heads
        q_channel = hidden_size // self.tp_group_size
        kv_channel = (hidden_size // n_rep) // self.tp_group_size
        q_weight = qkv_bias[: q_channel]
        k_weight = qkv_bias[q_channel: q_channel + kv_channel]
        v_weight = qkv_bias[q_channel + kv_channel: q_channel + 2 * kv_channel]
        q_w_reshape = q_weight.reshape(n_kv_heads // self.tp_group_size, hidden_size // n_kv_heads)
        k_w_reshape = k_weight.reshape(n_kv_heads // self.tp_group_size, hidden_size // num_heads)
        v_w_reshape = v_weight.reshape(n_kv_heads // self.tp_group_size, hidden_size // num_heads)

        cat_qkv_weight = np.concatenate((q_w_reshape, k_w_reshape, v_w_reshape), axis=1)
        out_qkv_weight = cat_qkv_weight.reshape(w,)
        return out_qkv_weight

    def ffn_concat_hf2mg(self, ffn_weights: np.ndarray, ffn_hidden_size):
        """
            convert ffn_concat weight with huggingface format to megatron format.
        """
        w, h = ffn_weights.shape
        gate_weight = ffn_weights[: w // 2, :]
        hidden_weight = ffn_weights[w // 2: w // 2 * 2, :]
        gate_w_reshape = gate_weight.reshape(-1, 1, ffn_hidden_size)
        hidden_w_reshape = hidden_weight.reshape(-1, 1, ffn_hidden_size)
        cat_ffn_weight = np.concatenate((gate_w_reshape, hidden_w_reshape), axis=1)
        out_ffn_weight = cat_ffn_weight.reshape(w, h)
        return out_ffn_weight

    def ffn_quant_params_concat_hf2mg(self, gate_params: np.ndarray, hidden_params: np.ndarray,):
        """
            convert ffn_concat quant params with huggingface format to megatron format.
        """
        w = gate_params.shape[0] + hidden_params.shape[0]
        gate_weight_channel = gate_params.shape[0]
        hidden_weight_channel = hidden_params.shape[0]
        gate_w_reshape = gate_params.reshape(gate_weight_channel, 1)
        hidden_w_reshape = hidden_params.reshape(hidden_weight_channel, 1)
        cat_ffn_weight = np.concatenate((gate_w_reshape, hidden_w_reshape), axis=1)
        out_ffn_weight = cat_ffn_weight.reshape(w, )
        return out_ffn_weight

    def infer_convert_outer_weight(self, src_hf_dir, hf_weight_map):
        """convert weight not in model"""
        embed_tokens_hf_name = "model.embed_tokens.weight"
        embed_tokens_ms_name = self.convert_weight_name(embed_tokens_hf_name)
        if self.config.parallel_config.vocab_emb_dp:
            np_data, _ = self.get_safetensor_from_file(embed_tokens_hf_name, src_hf_dir, hf_weight_map)
        else:
            np_data, _ = self.get_safetensor_from_file(embed_tokens_hf_name, src_hf_dir, hf_weight_map,
                                                       is_split_param=True, split_axis=0)
        embed_tokens_dtype = convert_np_to_ms_dtype(np_data)
        self.parameter_dict[embed_tokens_ms_name] = ms.Parameter(ms.from_numpy(np_data).astype(embed_tokens_dtype),
                                                                 name=embed_tokens_ms_name,
                                                                 requires_grad=False)

        norm_hf_name = "model.norm.weight"
        norm_ms_name = self.convert_weight_name(norm_hf_name)
        np_data, _ = self.get_safetensor_from_file(norm_hf_name, src_hf_dir, hf_weight_map)
        norm_dtype = convert_np_to_ms_dtype(np_data)
        self.parameter_dict[norm_ms_name] = ms.Parameter(ms.from_numpy(np_data).astype(norm_dtype),
                                                         name=norm_ms_name,
                                                         requires_grad=False)

        lm_head_hf_name = "lm_head.weight"
        lm_head_ms_name = self.convert_weight_name(lm_head_hf_name)
        if not self.config.model.model_config.tie_word_embeddings:
            if not self.config.parallel_config.vocab_emb_dp:
                np_data, _ = self.get_safetensor_from_file(lm_head_hf_name, src_hf_dir, hf_weight_map,
                                                           is_split_param=True, split_axis=0)
            else:
                np_data, _ = self.get_safetensor_from_file(lm_head_hf_name, src_hf_dir, hf_weight_map)
            lm_head_dtype = convert_np_to_ms_dtype(np_data)
            self.parameter_dict[lm_head_ms_name] = ms.Parameter(ms.from_numpy(np_data).astype(lm_head_dtype),
                                                                name=lm_head_ms_name,
                                                                requires_grad=False)

    def convert_weight_name(self, weight_name: str):
        """replace weight name"""
        weight_name = weight_name.replace('embed_tokens.weight', 'tok_embeddings.embedding_weight')
        weight_name = weight_name.replace('self_attn.q_proj.', 'attention.wq.')
        weight_name = weight_name.replace('self_attn.k_proj.', 'attention.wk.')
        weight_name = weight_name.replace('self_attn.v_proj.', 'attention.wv.')
        weight_name = weight_name.replace('self_attn.o_proj.', 'attention.wo.')

        weight_name = weight_name.replace('mlp.gate_proj.', 'feed_forward.w1.')
        weight_name = weight_name.replace('mlp.down_proj.', 'feed_forward.w2.')
        weight_name = weight_name.replace('mlp.up_proj.', 'feed_forward.w3.')
        weight_name = weight_name.replace('.input_layernorm.', '.attention_norm.')
        weight_name = weight_name.replace('.post_attention_layernorm.', '.ffn_norm.')
        weight_name = weight_name.replace('model.norm.weight', 'model.norm_out.weight')
        return weight_name

    def convert_quant_weight_name(self, weight_name: str):
        """replace weight name"""
        weight_name = weight_name.replace('embed_tokens.weight', 'tok_embeddings.embedding_weight')

        weight_name = weight_name.replace('.self_attn.q_proj.weight', '.attention.wq._layer.weight')
        weight_name = weight_name.replace('.self_attn.q_proj.bias', '.attention.wq._layer.bias')
        weight_name = weight_name.replace('.self_attn.q_proj.input_scale', '.attention.wq.quant_op.input_scale')
        weight_name = weight_name.replace('.self_attn.q_proj.input_offset', '.attention.wq.quant_op.input_zp')
        weight_name = weight_name.replace('.self_attn.q_proj.quant_bias', '.attention.wq._layer.matmul.quant_bias')
        weight_name = weight_name.replace('.self_attn.q_proj.deq_scale', '.attention.wq._layer.matmul.dequant_scale')

        weight_name = weight_name.replace('.self_attn.k_proj.weight', '.attention.wk._layer.weight')
        weight_name = weight_name.replace('.self_attn.k_proj.bias', '.attention.wk._layer.bias')
        weight_name = weight_name.replace('.self_attn.k_proj.input_scale', '.attention.wk.quant_op.input_scale')
        weight_name = weight_name.replace('.self_attn.k_proj.input_offset', '.attention.wk.quant_op.input_zp')
        weight_name = weight_name.replace('.self_attn.k_proj.quant_bias', '.attention.wk._layer.matmul.quant_bias')
        weight_name = weight_name.replace('.self_attn.k_proj.deq_scale', '.attention.wk._layer.matmul.dequant_scale')

        weight_name = weight_name.replace('.self_attn.v_proj.weight', '.attention.wv._layer.weight')
        weight_name = weight_name.replace('.self_attn.v_proj.bias', '.attention.wv._layer.bias')
        weight_name = weight_name.replace('.self_attn.v_proj.input_scale', '.attention.wv.quant_op.input_scale')
        weight_name = weight_name.replace('.self_attn.v_proj.input_offset', '.attention.wv.quant_op.input_zp')
        weight_name = weight_name.replace('.self_attn.v_proj.quant_bias', '.attention.wv._layer.matmul.quant_bias')
        weight_name = weight_name.replace('.self_attn.v_proj.deq_scale', '.attention.wv._layer.matmul.dequant_scale')

        weight_name = weight_name.replace('.self_attn.o_proj.weight', '.attention.wo._layer.weight')
        weight_name = weight_name.replace('.self_attn.o_proj.bias', '.attention.wo._layer.bias')
        weight_name = weight_name.replace('.self_attn.o_proj.input_scale', '.attention.wo.quant_op.input_scale')
        weight_name = weight_name.replace('.self_attn.o_proj.input_offset', '.attention.wo.quant_op.input_zp')
        weight_name = weight_name.replace('.self_attn.o_proj.quant_bias', '.attention.wo._layer.matmul.quant_bias')
        weight_name = weight_name.replace('.self_attn.o_proj.deq_scale', '.attention.wo._layer.matmul.dequant_scale')

        weight_name = weight_name.replace('.mlp.gate_proj.weight', '.feed_forward.w1._layer.weight')
        weight_name = weight_name.replace('.mlp.gate_proj.bias', '.feed_forward.w1._layer.bias')
        weight_name = weight_name.replace('.mlp.gate_proj.input_scale', '.feed_forward.w1.quant_op.input_scale')
        weight_name = weight_name.replace('.mlp.gate_proj.input_offset', '.feed_forward.w1.quant_op.input_zp')
        weight_name = weight_name.replace('.mlp.gate_proj.quant_bias', '.feed_forward.w1._layer.matmul.quant_bias')
        weight_name = weight_name.replace('.mlp.gate_proj.deq_scale', '.feed_forward.w1._layer.matmul.dequant_scale')

        weight_name = weight_name.replace('.mlp.down_proj.weight', '.feed_forward.w2.weight')

        weight_name = weight_name.replace('.mlp.up_proj.weight', '.feed_forward.w3._layer.weight')
        weight_name = weight_name.replace('.mlp.up_proj.bias', '.feed_forward.w3._layer.bias')
        weight_name = weight_name.replace('.mlp.up_proj.input_scale', '.feed_forward.w3.quant_op.input_scale')
        weight_name = weight_name.replace('.mlp.up_proj.input_offset', '.feed_forward.w3.quant_op.input_zp')
        weight_name = weight_name.replace('.mlp.up_proj.quant_bias', '.feed_forward.w3._layer.matmul.quant_bias')
        weight_name = weight_name.replace('.mlp.up_proj.deq_scale', '.feed_forward.w3._layer.matmul.dequant_scale')

        weight_name = weight_name.replace('.input_layernorm.', '.attention_norm.')
        weight_name = weight_name.replace('.post_attention_layernorm.', '.ffn_norm.')
        weight_name = weight_name.replace('model.norm.weight', 'model.norm_out.weight')
        return weight_name

    def infer_process_dense_ffn_weight(self, src_hf_dir, layer_id, hf_weight_map):
        """infer process dense ffn weight"""

        ffn_concat = self.config.model.model_config.ffn_concat
        w1_hf_name = f"model.layers.{layer_id}.mlp.gate_proj.weight"
        w1_ms_name = self.convert_weight_name(w1_hf_name)
        w1_ms_param, _ = self.get_safetensor_from_file(w1_hf_name, src_hf_dir, hf_weight_map, is_split_param=True,
                                                       split_axis=0)

        w2_hf_name = f"model.layers.{layer_id}.mlp.down_proj.weight"
        w2_ms_name = self.convert_weight_name(w2_hf_name)
        w2_ms_param, _ = self.get_safetensor_from_file(w2_hf_name, src_hf_dir, hf_weight_map, is_split_param=True,
                                                       split_axis=1)

        w3_hf_name = f"model.layers.{layer_id}.mlp.up_proj.weight"
        w3_ms_name = self.convert_weight_name(w3_hf_name)
        w3_ms_param, _ = self.get_safetensor_from_file(w3_hf_name, src_hf_dir, hf_weight_map, is_split_param=True,
                                                       split_axis=0)

        weight_dtype = convert_np_to_ms_dtype(w1_ms_param)

        if ffn_concat:
            w_gate_hidden_name = f"model.layers.{layer_id}.feed_forward.w_gate_hidden.weight"
            w_gate_hidden_param = np.concatenate((w1_ms_param, w3_ms_param), axis=0)
            w_gate_hidden_param = self.ffn_concat_hf2mg(w_gate_hidden_param, self.hidden_size)
            w_gate_hidden_param = ms.from_numpy(w_gate_hidden_param).astype(weight_dtype)
            self.parameter_dict[w_gate_hidden_name] = ms.Parameter(w_gate_hidden_param, name=w_gate_hidden_name,
                                                                   requires_grad=False)
        else:
            self.parameter_dict[w1_ms_name] = ms.Parameter(ms.from_numpy(w1_ms_param).astype(weight_dtype),
                                                           name=w1_ms_name,
                                                           requires_grad=False)
            self.parameter_dict[w3_ms_name] = ms.Parameter(ms.from_numpy(w3_ms_param).astype(weight_dtype),
                                                           name=w3_ms_name,
                                                           requires_grad=False)

        self.parameter_dict[w2_ms_name] = ms.Parameter(ms.from_numpy(w2_ms_param).astype(weight_dtype),
                                                       name=w2_ms_name,
                                                       requires_grad=False)

    def infer_process_attention_weight(self, src_hf_dir, layer_id, hf_weight_map):
        """infer process attention weight"""
        qkv_concat = self.config.model.model_config.qkv_concat
        # wq
        wq_hf_name = f"model.layers.{layer_id}.self_attn.q_proj.weight"
        wq_ms_name = self.convert_weight_name(wq_hf_name)
        wq_ms_param, _ = self.get_safetensor_from_file(wq_hf_name, src_hf_dir, hf_weight_map, is_split_param=True,
                                                       split_axis=0)
        # wq bias
        wq_bias_hf_name = f"model.layers.{layer_id}.self_attn.q_proj.bias"
        wq_bias_ms_name = self.convert_weight_name(wq_bias_hf_name)
        wq_bias_ms_param, _ = self.get_safetensor_from_file(wq_bias_hf_name, src_hf_dir, hf_weight_map,
                                                            is_split_param=True,
                                                            split_axis=0)

        # wk
        wk_hf_name = f"model.layers.{layer_id}.self_attn.k_proj.weight"
        wk_ms_name = self.convert_weight_name(wk_hf_name)
        wk_ms_param, _ = self.get_safetensor_from_file(wk_hf_name, src_hf_dir, hf_weight_map, is_split_param=True,
                                                       split_axis=0)
        # wk bias
        wk_bias_hf_name = f"model.layers.{layer_id}.self_attn.k_proj.bias"
        wk_bias_ms_name = self.convert_weight_name(wk_bias_hf_name)
        wk_bias_ms_param, _ = self.get_safetensor_from_file(wk_bias_hf_name, src_hf_dir, hf_weight_map,
                                                            is_split_param=True,
                                                            split_axis=0)

        # wv
        wv_hf_name = f"model.layers.{layer_id}.self_attn.v_proj.weight"
        wv_ms_name = self.convert_weight_name(wv_hf_name)
        wv_ms_param, _ = self.get_safetensor_from_file(wv_hf_name, src_hf_dir, hf_weight_map, is_split_param=True,
                                                       split_axis=0)
        # wv bias
        wv_bias_hf_name = f"model.layers.{layer_id}.self_attn.v_proj.bias"
        wv_bias_ms_name = self.convert_weight_name(wv_bias_hf_name)
        wv_bias_ms_param, _ = self.get_safetensor_from_file(wv_bias_hf_name, src_hf_dir, hf_weight_map,
                                                            is_split_param=True,
                                                            split_axis=0)

        weight_dtype = convert_np_to_ms_dtype(wq_ms_param)
        bias_dtype = convert_np_to_ms_dtype(wq_bias_ms_param)

        if qkv_concat:
            w_qkv_name = f"model.layers.{layer_id}.attention.w_qkv.weight"
            w_qkv_param = np.concatenate((wq_ms_param, wk_ms_param, wv_ms_param), axis=0)
            w_qkv_param = self.qkv_concat_hf2mg(w_qkv_param, self.num_heads, self.kv_heads, self.hidden_size)
            w_qkv_param = ms.from_numpy(w_qkv_param).astype(weight_dtype)
            self.parameter_dict[w_qkv_name] = ms.Parameter(w_qkv_param, name=w_qkv_name, requires_grad=False)

            w_qkv_bias_name = f"model.layers.{layer_id}.attention.w_qkv.bias"
            w_qkv_bias_param = np.concatenate((wq_bias_ms_param, wk_bias_ms_param, wv_bias_ms_param), axis=0)
            w_qkv_bias_param = self.qkv_bias_concat_hf2mg(w_qkv_bias_param, self.num_heads, self.kv_heads, self.hidden_size)
            w_qkv_bias_param = ms.from_numpy(w_qkv_bias_param).astype(bias_dtype)
            self.parameter_dict[w_qkv_bias_name] = ms.Parameter(w_qkv_bias_param, name=w_qkv_bias_name,
                                                                requires_grad=False)
        else:
            self.parameter_dict[wq_ms_name] = ms.Parameter(ms.from_numpy(wq_ms_param).astype(weight_dtype),
                                                           name=wq_ms_name,
                                                           requires_grad=False)
            self.parameter_dict[wk_ms_name] = ms.Parameter(ms.from_numpy(wk_ms_param).astype(weight_dtype),
                                                           name=wk_ms_name,
                                                           requires_grad=False)
            self.parameter_dict[wv_ms_name] = ms.Parameter(ms.from_numpy(wv_ms_param).astype(weight_dtype),
                                                           name=wv_ms_name,
                                                           requires_grad=False)
            self.parameter_dict[wq_bias_ms_name] = ms.Parameter(
                ms.from_numpy(wq_bias_ms_param).astype(bias_dtype),
                name=wq_bias_ms_name,
                requires_grad=False)
            self.parameter_dict[wk_bias_ms_name] = ms.Parameter(
                ms.from_numpy(wk_bias_ms_param).astype(bias_dtype),
                name=wk_bias_ms_name,
                requires_grad=False)
            self.parameter_dict[wv_bias_ms_name] = ms.Parameter(
                ms.from_numpy(wv_bias_ms_param).astype(bias_dtype),
                name=wv_bias_ms_name,
                requires_grad=False)

        # wo
        wo_hf_name = f"model.layers.{layer_id}.self_attn.o_proj.weight"
        wo_ms_name = self.convert_weight_name(wo_hf_name)
        wo_ms_param, _ = self.get_safetensor_from_file(wo_hf_name, src_hf_dir, hf_weight_map, is_split_param=True,
                                                       split_axis=1)
        self.parameter_dict[wo_ms_name] = ms.Parameter(ms.from_numpy(wo_ms_param).astype(weight_dtype),
                                                       name=wo_ms_name,
                                                       requires_grad=False)

    def infer_process_quant_attention_weight(self, src_hf_dir, layer_id, hf_weight_map):
        """infer process quant attention weight"""
        qkv_concat = self.config.model.model_config.qkv_concat
        # wq weight
        wq_hf_name = f"model.layers.{layer_id}.self_attn.q_proj.weight"
        wq_ms_name = self.convert_quant_weight_name(wq_hf_name)
        wq_ms_param, _ = self.get_safetensor_from_file(wq_hf_name, src_hf_dir, hf_weight_map, is_split_param=True,
                                                       split_axis=0)

        # wq quant params
        wq_hf_quant_scale_name = f"model.layers.{layer_id}.self_attn.q_proj.input_scale"
        wq_ms_quant_scale_name = self.convert_quant_weight_name(wq_hf_quant_scale_name)
        wq_ms_quant_scale_param, _ = self.get_safetensor_from_file(wq_hf_quant_scale_name, src_hf_dir, hf_weight_map)
        wq_hf_quant_zp_name = f"model.layers.{layer_id}.self_attn.q_proj.input_offset"
        wq_ms_quant_zp_name = self.convert_quant_weight_name(wq_hf_quant_zp_name)
        wq_ms_quant_zp_param, _ = self.get_safetensor_from_file(wq_hf_quant_zp_name, src_hf_dir, hf_weight_map)
        wq_hf_scale_name = f"model.layers.{layer_id}.self_attn.q_proj.deq_scale"
        wq_ms_scale_name = self.convert_quant_weight_name(wq_hf_scale_name)
        wq_ms_scale_param, _ = self.get_safetensor_from_file(wq_hf_scale_name, src_hf_dir, hf_weight_map,
                                                             is_split_param=True,
                                                             split_axis=0)
        wq_hf_quant_bias_name = f"model.layers.{layer_id}.self_attn.q_proj.quant_bias"
        wq_ms_quant_bias_name = self.convert_quant_weight_name(wq_hf_quant_bias_name)
        wq_ms_quant_bias_param, _ = self.get_safetensor_from_file(wq_hf_quant_bias_name, src_hf_dir, hf_weight_map,
                                                                  is_split_param=True,
                                                                  split_axis=0)

        # wq bias
        wq_bias_hf_name = f"model.layers.{layer_id}.self_attn.q_proj.bias"
        wq_bias_ms_name = self.convert_quant_weight_name(wq_bias_hf_name)
        wq_bias_ms_param, _ = self.get_safetensor_from_file(wq_bias_hf_name, src_hf_dir, hf_weight_map,
                                                            is_split_param=True,
                                                            split_axis=0)

        # wk
        wk_hf_name = f"model.layers.{layer_id}.self_attn.k_proj.weight"
        wk_ms_name = self.convert_quant_weight_name(wk_hf_name)
        wk_ms_param, _ = self.get_safetensor_from_file(wk_hf_name, src_hf_dir, hf_weight_map, is_split_param=True,
                                                       split_axis=0)

        # wk quant params
        wk_hf_quant_scale_name = f"model.layers.{layer_id}.self_attn.k_proj.input_scale"
        wk_ms_quant_scale_name = self.convert_quant_weight_name(wk_hf_quant_scale_name)
        wk_ms_quant_scale_param, _ = self.get_safetensor_from_file(wk_hf_quant_scale_name, src_hf_dir, hf_weight_map)
        wk_hf_quant_zp_name = f"model.layers.{layer_id}.self_attn.k_proj.input_offset"
        wk_ms_quant_zp_name = self.convert_quant_weight_name(wk_hf_quant_zp_name)
        wk_ms_quant_zp_param, _ = self.get_safetensor_from_file(wk_hf_quant_zp_name, src_hf_dir, hf_weight_map)
        wk_hf_scale_name = f"model.layers.{layer_id}.self_attn.k_proj.deq_scale"
        wk_ms_scale_name = self.convert_quant_weight_name(wk_hf_scale_name)
        wk_ms_scale_param, _ = self.get_safetensor_from_file(wk_hf_scale_name, src_hf_dir, hf_weight_map,
                                                             is_split_param=True,
                                                             split_axis=0)
        wk_hf_quant_bias_name = f"model.layers.{layer_id}.self_attn.k_proj.quant_bias"
        wk_ms_quant_bias_name = self.convert_quant_weight_name(wk_hf_quant_bias_name)
        wk_ms_quant_bias_param, _ = self.get_safetensor_from_file(wk_hf_quant_bias_name, src_hf_dir, hf_weight_map,
                                                                  is_split_param=True,
                                                                  split_axis=0)

        # wk bias
        wk_bias_hf_name = f"model.layers.{layer_id}.self_attn.k_proj.bias"
        wk_bias_ms_name = self.convert_quant_weight_name(wk_bias_hf_name)
        wk_bias_ms_param, _ = self.get_safetensor_from_file(wk_bias_hf_name, src_hf_dir, hf_weight_map,
                                                            is_split_param=True,
                                                            split_axis=0)

        # wv
        wv_hf_name = f"model.layers.{layer_id}.self_attn.v_proj.weight"
        wv_ms_name = self.convert_quant_weight_name(wv_hf_name)
        wv_ms_param, _ = self.get_safetensor_from_file(wv_hf_name, src_hf_dir, hf_weight_map, is_split_param=True,
                                                       split_axis=0)

        # wv quant params
        wv_hf_quant_scale_name = f"model.layers.{layer_id}.self_attn.v_proj.input_scale"
        wv_ms_quant_scale_name = self.convert_quant_weight_name(wv_hf_quant_scale_name)
        wv_ms_quant_scale_param, _ = self.get_safetensor_from_file(wv_hf_quant_scale_name, src_hf_dir, hf_weight_map)
        wv_hf_quant_zp_name = f"model.layers.{layer_id}.self_attn.v_proj.input_offset"
        wv_ms_quant_zp_name = self.convert_quant_weight_name(wv_hf_quant_zp_name)
        wv_ms_quant_zp_param, _ = self.get_safetensor_from_file(wv_hf_quant_zp_name, src_hf_dir, hf_weight_map)
        wv_hf_scale_name = f"model.layers.{layer_id}.self_attn.v_proj.deq_scale"
        wv_ms_scale_name = self.convert_quant_weight_name(wv_hf_scale_name)
        wv_ms_scale_param, _ = self.get_safetensor_from_file(wv_hf_scale_name, src_hf_dir, hf_weight_map,
                                                             is_split_param=True,
                                                             split_axis=0)
        wv_hf_quant_bias_name = f"model.layers.{layer_id}.self_attn.v_proj.quant_bias"
        wv_ms_quant_bias_name = self.convert_quant_weight_name(wv_hf_quant_bias_name)
        wv_ms_quant_bias_param, _ = self.get_safetensor_from_file(wv_hf_quant_bias_name, src_hf_dir, hf_weight_map,
                                                                  is_split_param=True,
                                                                  split_axis=0)

        # wv bias
        wv_bias_hf_name = f"model.layers.{layer_id}.self_attn.v_proj.bias"
        wv_bias_ms_name = self.convert_quant_weight_name(wv_bias_hf_name)
        wv_bias_ms_param, _ = self.get_safetensor_from_file(wv_bias_hf_name, src_hf_dir, hf_weight_map,
                                                            is_split_param=True,
                                                            split_axis=0)

        if qkv_concat:
            w_qkv_name = f"model.layers.{layer_id}.attention.w_qkv._layer.weight"
            w_qkv_param = np.concatenate((wq_ms_param, wk_ms_param, wv_ms_param), axis=0)
            w_qkv_param = ms.from_numpy(w_qkv_param).astype(ms.int8)
            w_qkv_param = self.qkv_concat_hf2mg(w_qkv_param, self.num_heads, self.kv_heads, self.hidden_size)
            self.parameter_dict[w_qkv_name] = ms.Parameter(w_qkv_param, name=w_qkv_name, requires_grad=False)

            w_qkv_quant_scale_name = f"model.layers.{layer_id}.attention.w_qkv.quant_op.input_scale"
            w_qkv_quant_scale_param = ms.from_numpy(wq_ms_quant_scale_param).astype(ms.float16)
            w_qkv_quant_scale_param = np.full(shape=w_qkv_param.shape[-1], fill_value=w_qkv_quant_scale_param.item())
            self.parameter_dict[w_qkv_quant_scale_name] = ms.Parameter(w_qkv_quant_scale_param,
                                                                       name=w_qkv_quant_scale_name, requires_grad=False)

            w_qkv_quant_zp_name = f"model.layers.{layer_id}.attention.w_qkv.quant_op.input_zp"
            w_qkv_quant_zp_param = ms.from_numpy(wq_ms_quant_zp_param).astype(ms.int8)
            w_qkv_quant_zp_param = np.full(shape=w_qkv_param.shape[-1], fill_value=w_qkv_quant_zp_param.item())
            self.parameter_dict[w_qkv_quant_zp_name] = ms.Parameter(w_qkv_quant_zp_param, name=w_qkv_quant_zp_name,
                                                                    requires_grad=False)

            w_qkv_quant_bias_name = f"model.layers.{layer_id}.attention.w_qkv._layer.matmul.quant_bias"
            w_qkv_quant_bias_param = np.concatenate(
                (wq_ms_quant_bias_param, wk_ms_quant_bias_param, wv_ms_quant_bias_param), axis=0)
            w_qkv_quant_bias_param = self.qkv_bias_concat_hf2mg(w_qkv_quant_bias_param, self.num_heads, self.kv_heads,
                                                                self.hidden_size)
            w_qkv_quant_bias_param = ms.from_numpy(w_qkv_quant_bias_param).astype(ms.int32)
            self.parameter_dict[w_qkv_quant_bias_name] = ms.Parameter(w_qkv_quant_bias_param,
                                                                      name=w_qkv_quant_bias_name, requires_grad=False)

            w_qkv_scale_name = f"model.layers.{layer_id}.attention.w_qkv._layer.matmul.dequant_scale"
            w_qkv_scale_param = np.concatenate((wq_ms_scale_param, wk_ms_scale_param, wv_ms_scale_param), axis=0)
            w_qkv_scale_param = self.qkv_bias_concat_hf2mg(w_qkv_scale_param, self.num_heads, self.kv_heads,
                                                           self.hidden_size)
            w_qkv_scale_param = np.frombuffer(w_qkv_scale_param.astype(np.float32).tobytes(), dtype=np.int32).astype(
                np.int64)
            w_qkv_scale_param = ms.from_numpy(w_qkv_scale_param).astype(ms.int64)
            self.parameter_dict[w_qkv_scale_name] = ms.Parameter(w_qkv_scale_param, name=w_qkv_scale_name,
                                                                 requires_grad=False)

            w_qkv_bias_name = f"model.layers.{layer_id}.attention.w_qkv._layer.bias"
            w_qkv_bias_param = np.concatenate((wq_bias_ms_param, wk_bias_ms_param, wv_bias_ms_param), axis=0)
            w_qkv_bias_param = ms.from_numpy(w_qkv_bias_param).astype(ms.float16)
            w_qkv_bias_param = self.qkv_bias_concat_hf2mg(w_qkv_bias_param, self.num_heads, self.kv_heads,
                                                          self.hidden_size)
            self.parameter_dict[w_qkv_bias_name] = ms.Parameter(w_qkv_bias_param, name=w_qkv_bias_name,
                                                                requires_grad=False)
        else:
            self.parameter_dict[wq_ms_name] = ms.Parameter(ms.from_numpy(wq_ms_param).astype(ms.int8),
                                                           name=wq_ms_name,
                                                           requires_grad=False)
            wq_ms_quant_scale_param = np.full(shape=wq_ms_param.shape[-1], fill_value=wq_ms_quant_scale_param.item())
            self.parameter_dict[wq_ms_quant_scale_name] = ms.Parameter(
                ms.from_numpy(wq_ms_quant_scale_param).astype(ms.float16),
                name=wq_ms_name,
                requires_grad=False)
            wq_ms_quant_zp_param = np.full(shape=wq_ms_param.shape[-1], fill_value=wq_ms_quant_zp_param.item())
            self.parameter_dict[wq_ms_quant_zp_name] = ms.Parameter(ms.from_numpy(wq_ms_quant_zp_param).astype(ms.int8),
                                                                    name=wq_ms_name,
                                                                    requires_grad=False)
            self.parameter_dict[wq_ms_quant_bias_name] = ms.Parameter(
                ms.from_numpy(wq_ms_quant_bias_param).astype(ms.int32),
                name=wq_ms_name,
                requires_grad=False)
            wq_ms_scale_param = np.frombuffer(wq_ms_scale_param.astype(np.float32).tobytes(), dtype=np.int32).astype(
                np.int64)
            self.parameter_dict[wq_ms_scale_name] = ms.Parameter(ms.from_numpy(wq_ms_scale_param).astype(ms.int64),
                                                                 name=wq_ms_name,
                                                                 requires_grad=False)

            self.parameter_dict[wk_ms_name] = ms.Parameter(ms.from_numpy(wk_ms_param).astype(ms.int8),
                                                           name=wk_ms_name,
                                                           requires_grad=False)
            wk_ms_quant_scale_param = np.full(shape=wk_ms_param.shape[-1], fill_value=wk_ms_quant_scale_param.item())
            self.parameter_dict[wk_ms_quant_scale_name] = ms.Parameter(
                ms.from_numpy(wk_ms_quant_scale_param).astype(ms.float16),
                name=wq_ms_name,
                requires_grad=False)
            wk_ms_quant_zp_param = np.full(shape=wk_ms_param.shape[-1], fill_value=wk_ms_quant_zp_param.item())
            self.parameter_dict[wk_ms_quant_zp_name] = ms.Parameter(ms.from_numpy(wk_ms_quant_zp_param).astype(ms.int8),
                                                                    name=wq_ms_name,
                                                                    requires_grad=False)
            self.parameter_dict[wk_ms_quant_bias_name] = ms.Parameter(
                ms.from_numpy(wk_ms_quant_bias_param).astype(ms.int32),
                name=wq_ms_name,
                requires_grad=False)
            wk_ms_scale_param = np.frombuffer(wk_ms_scale_param.astype(np.float32).tobytes(), dtype=np.int32).astype(
                np.int64)
            self.parameter_dict[wk_ms_scale_name] = ms.Parameter(ms.from_numpy(wk_ms_scale_param).astype(ms.int64),
                                                                 name=wq_ms_name,
                                                                 requires_grad=False)

            self.parameter_dict[wv_ms_name] = ms.Parameter(ms.from_numpy(wv_ms_param).astype(ms.int8),
                                                           name=wv_ms_name,
                                                           requires_grad=False)
            wv_ms_quant_scale_param = np.full(shape=wv_ms_param.shape[-1], fill_value=wv_ms_quant_scale_param.item())
            self.parameter_dict[wv_ms_quant_scale_name] = ms.Parameter(
                ms.from_numpy(wv_ms_quant_scale_param).astype(ms.float16),
                name=wq_ms_name,
                requires_grad=False)
            wv_ms_quant_zp_param = np.full(shape=wv_ms_param.shape[-1], fill_value=wv_ms_quant_zp_param.item())
            self.parameter_dict[wv_ms_quant_zp_name] = ms.Parameter(ms.from_numpy(wv_ms_quant_zp_param).astype(ms.int8),
                                                                    name=wq_ms_name,
                                                                    requires_grad=False)
            self.parameter_dict[wv_ms_quant_bias_name] = ms.Parameter(
                ms.from_numpy(wv_ms_quant_bias_param).astype(ms.int32),
                name=wq_ms_name,
                requires_grad=False)
            wv_ms_scale_param = np.frombuffer(wv_ms_scale_param.astype(np.float32).tobytes(), dtype=np.int32).astype(
                np.int64)
            self.parameter_dict[wv_ms_scale_name] = ms.Parameter(ms.from_numpy(wv_ms_scale_param).astype(ms.int64),
                                                                 name=wq_ms_name,
                                                                 requires_grad=False)

            self.parameter_dict[wq_bias_ms_name] = ms.Parameter(
                ms.from_numpy(wq_bias_ms_param).astype(ms.float16),
                name=wq_bias_ms_name,
                requires_grad=False)
            self.parameter_dict[wk_bias_ms_name] = ms.Parameter(
                ms.from_numpy(wk_bias_ms_param).astype(ms.float16),
                name=wk_bias_ms_name,
                requires_grad=False)
            self.parameter_dict[wv_bias_ms_name] = ms.Parameter(
                ms.from_numpy(wv_bias_ms_param).astype(ms.float16),
                name=wv_bias_ms_name,
                requires_grad=False)

        # wo
        wo_hf_name = f"model.layers.{layer_id}.self_attn.o_proj.weight"
        wo_ms_name = self.convert_quant_weight_name(wo_hf_name)
        wo_ms_param, _ = self.get_safetensor_from_file(wo_hf_name, src_hf_dir, hf_weight_map, is_split_param=True,
                                                       split_axis=1)
        # wo quant params
        wo_hf_quant_scale_name = f"model.layers.{layer_id}.self_attn.o_proj.input_scale"
        wo_ms_quant_scale_name = self.convert_quant_weight_name(wo_hf_quant_scale_name)
        wo_ms_quant_scale_param, _ = self.get_safetensor_from_file(wo_hf_quant_scale_name, src_hf_dir, hf_weight_map)
        wo_hf_quant_zp_name = f"model.layers.{layer_id}.self_attn.o_proj.input_offset"
        wo_ms_quant_zp_name = self.convert_quant_weight_name(wo_hf_quant_zp_name)
        wo_ms_quant_zp_param, _ = self.get_safetensor_from_file(wo_hf_quant_zp_name, src_hf_dir, hf_weight_map)
        wo_hf_scale_name = f"model.layers.{layer_id}.self_attn.o_proj.deq_scale"
        wo_ms_scale_name = self.convert_quant_weight_name(wo_hf_scale_name)
        wo_ms_scale_param, _ = self.get_safetensor_from_file(wo_hf_scale_name, src_hf_dir, hf_weight_map)
        wo_hf_quant_bias_name = f"model.layers.{layer_id}.self_attn.o_proj.quant_bias"
        wo_ms_quant_bias_name = self.convert_quant_weight_name(wo_hf_quant_bias_name)
        wo_ms_quant_bias_param, _ = self.get_safetensor_from_file(wo_hf_quant_bias_name, src_hf_dir, hf_weight_map)
        self.parameter_dict[wo_ms_name] = ms.Parameter(ms.from_numpy(wo_ms_param).astype(ms.int8),
                                                       name=wo_ms_name,
                                                       requires_grad=False)
        wo_ms_quant_scale_param = np.full(shape=wo_ms_param.shape[-1], fill_value=wo_ms_quant_scale_param.item())
        self.parameter_dict[wo_ms_quant_scale_name] = ms.Parameter(
            ms.from_numpy(wo_ms_quant_scale_param).astype(ms.float16),
            name=wo_ms_name,
            requires_grad=False)
        wo_ms_quant_zp_param = np.full(shape=wo_ms_param.shape[-1], fill_value=wo_ms_quant_zp_param.item())
        self.parameter_dict[wo_ms_quant_zp_name] = ms.Parameter(ms.from_numpy(wo_ms_quant_zp_param).astype(ms.int8),
                                                                name=wo_ms_name,
                                                                requires_grad=False)
        wo_ms_scale_param = np.frombuffer(wo_ms_scale_param.astype(np.float32).tobytes(), dtype=np.int32).astype(
            np.int64)
        self.parameter_dict[wo_ms_scale_name] = ms.Parameter(ms.from_numpy(wo_ms_scale_param).astype(ms.int64),
                                                             name=wo_ms_name,
                                                             requires_grad=False)
        self.parameter_dict[wo_ms_quant_bias_name] = ms.Parameter(
            ms.from_numpy(wo_ms_quant_bias_param).astype(ms.int32),
            name=wo_ms_name,
            requires_grad=False)

    def infer_process_quant_ffn_weight(self, src_hf_dir, layer_id, hf_weight_map):
        """infer process quant ffn weight"""

        ffn_concat = self.config.model.model_config.qkv_concat
        w1_hf_name = f"model.layers.{layer_id}.mlp.gate_proj.weight"
        w1_ms_name = self.convert_quant_weight_name(w1_hf_name)
        w1_ms_param, _ = self.get_safetensor_from_file(w1_hf_name, src_hf_dir, hf_weight_map, is_split_param=True,
                                                       split_axis=0)

        # w1 quant params
        w1_hf_quant_scale_name = f"model.layers.{layer_id}.mlp.gate_proj.input_scale"
        w1_ms_quant_scale_name = self.convert_quant_weight_name(w1_hf_quant_scale_name)
        w1_ms_quant_scale_param, _ = self.get_safetensor_from_file(w1_hf_quant_scale_name, src_hf_dir, hf_weight_map)

        w1_hf_quant_zp_name = f"model.layers.{layer_id}.mlp.gate_proj.input_offset"
        w1_ms_quant_zp_name = self.convert_quant_weight_name(w1_hf_quant_zp_name)
        w1_ms_quant_zp_param, _ = self.get_safetensor_from_file(w1_hf_quant_zp_name, src_hf_dir, hf_weight_map)
        w1_hf_scale_name = f"model.layers.{layer_id}.mlp.gate_proj.deq_scale"
        w1_ms_scale_name = self.convert_quant_weight_name(w1_hf_scale_name)
        w1_ms_scale_param, _ = self.get_safetensor_from_file(w1_hf_scale_name, src_hf_dir, hf_weight_map,
                                                             is_split_param=True,
                                                             split_axis=0)
        w1_hf_quant_bias_name = f"model.layers.{layer_id}.mlp.gate_proj.quant_bias"
        w1_ms_quant_bias_name = self.convert_quant_weight_name(w1_hf_quant_bias_name)
        w1_ms_quant_bias_param, _ = self.get_safetensor_from_file(w1_hf_quant_bias_name, src_hf_dir, hf_weight_map,
                                                                  is_split_param=True,
                                                                  split_axis=0)

        w2_hf_name = f"model.layers.{layer_id}.mlp.down_proj.weight"
        w2_ms_name = self.convert_quant_weight_name(w2_hf_name)
        w2_ms_param, _ = self.get_safetensor_from_file(w2_hf_name, src_hf_dir, hf_weight_map, is_split_param=True,
                                                       split_axis=1)

        w3_hf_name = f"model.layers.{layer_id}.mlp.up_proj.weight"
        w3_ms_name = self.convert_quant_weight_name(w3_hf_name)
        w3_ms_param, _ = self.get_safetensor_from_file(w3_hf_name, src_hf_dir, hf_weight_map, is_split_param=True,
                                                       split_axis=0)
        # w3 quant params
        w3_hf_quant_scale_name = f"model.layers.{layer_id}.mlp.up_proj.input_scale"
        w3_ms_quant_scale_name = self.convert_quant_weight_name(w3_hf_quant_scale_name)
        w3_ms_quant_scale_param, _ = self.get_safetensor_from_file(w3_hf_quant_scale_name, src_hf_dir, hf_weight_map)
        w3_hf_quant_zp_name = f"model.layers.{layer_id}.mlp.up_proj.input_offset"
        w3_ms_quant_zp_name = self.convert_quant_weight_name(w3_hf_quant_zp_name)
        w3_ms_quant_zp_param, _ = self.get_safetensor_from_file(w3_hf_quant_zp_name, src_hf_dir, hf_weight_map)
        w3_hf_scale_name = f"model.layers.{layer_id}.mlp.up_proj.deq_scale"
        w3_ms_scale_name = self.convert_quant_weight_name(w3_hf_scale_name)
        w3_ms_scale_param, _ = self.get_safetensor_from_file(w3_hf_scale_name, src_hf_dir, hf_weight_map,
                                                             is_split_param=True,
                                                             split_axis=0)
        w3_hf_quant_bias_name = f"model.layers.{layer_id}.mlp.up_proj.quant_bias"
        w3_ms_quant_bias_name = self.convert_quant_weight_name(w3_hf_quant_bias_name)
        w3_ms_quant_bias_param, _ = self.get_safetensor_from_file(w3_hf_quant_bias_name, src_hf_dir, hf_weight_map,
                                                                  is_split_param=True,
                                                                  split_axis=0)

        if ffn_concat:
            w_gate_hidden_name = f"model.layers.{layer_id}.feed_forward.w_gate_hidden._layer.weight"
            w_gate_hidden_param = np.concatenate((w1_ms_param, w3_ms_param), axis=0)
            w_gate_hidden_param = ms.from_numpy(w_gate_hidden_param).astype(ms.int8)
            w_gate_hidden_param = self.ffn_concat_hf2mg(w_gate_hidden_param, self.hidden_size)
            self.parameter_dict[w_gate_hidden_name] = ms.Parameter(w_gate_hidden_param, name=w_gate_hidden_name,
                                                                   requires_grad=False)
            w_gate_hidden_quant_scale_name = f"model.layers.{layer_id}.feed_forward.w_gate_hidden.quant_op.input_scale"
            w_gate_hidden_quant_scale_param = ms.from_numpy(w1_ms_quant_scale_param).astype(ms.float16)
            w_gate_hidden_quant_scale_param = np.full(shape=w_gate_hidden_param.shape[-1],
                                                      fill_value=w_gate_hidden_quant_scale_param.item())
            self.parameter_dict[w_gate_hidden_quant_scale_name] = ms.Parameter(w_gate_hidden_quant_scale_param,
                                                                               name=w_gate_hidden_quant_scale_name,
                                                                               requires_grad=False)
            w_gate_hidden_quant_zp_name = f"model.layers.{layer_id}.feed_forward.w_gate_hidden.quant_op.input_zp"
            w_gate_hidden_quant_zp_param = ms.from_numpy(w1_ms_quant_zp_param).astype(ms.int8)
            w_gate_hidden_quant_zp_param = np.full(shape=w_gate_hidden_param.shape[-1],
                                                      fill_value=w_gate_hidden_quant_zp_param.item())
            self.parameter_dict[w_gate_hidden_quant_zp_name] = ms.Parameter(w_gate_hidden_quant_zp_param,
                                                                            name=w_gate_hidden_quant_zp_name,
                                                                            requires_grad=False)
            w_gate_hidden_quant_bias_name = f"model.layers.{layer_id}.feed_forward.w_gate_hidden._layer.matmul.quant_bias"
            w_gate_hidden_quant_bias_param = self.ffn_quant_params_concat_hf2mg(w1_ms_quant_bias_param,
                                                                                w3_ms_quant_bias_param)
            w_gate_hidden_quant_bias_param = ms.from_numpy(w_gate_hidden_quant_bias_param).astype(ms.int32)
            self.parameter_dict[w_gate_hidden_quant_bias_name] = ms.Parameter(w_gate_hidden_quant_bias_param,
                                                                              name=w_gate_hidden_quant_bias_name,
                                                                              requires_grad=False)
            w_gate_hidden_scale_name = f"model.layers.{layer_id}.feed_forward.w_gate_hidden._layer.matmul.dequant_scale"
            w_gate_hidden_scale_param = self.ffn_quant_params_concat_hf2mg(w1_ms_scale_param, w3_ms_scale_param)
            w_gate_hidden_scale_param = np.frombuffer(w_gate_hidden_scale_param.astype(np.float32).tobytes(),
                                                      dtype=np.int32).astype(np.int64)
            w_gate_hidden_scale_param = ms.from_numpy(w_gate_hidden_scale_param).astype(ms.int64)
            self.parameter_dict[w_gate_hidden_scale_name] = ms.Parameter(w_gate_hidden_scale_param,
                                                                         name=w_gate_hidden_scale_name,
                                                                         requires_grad=False)
        else:
            self.parameter_dict[w1_ms_name] = ms.Parameter(ms.from_numpy(w1_ms_param).astype(ms.int8),
                                                           name=w1_ms_name,
                                                           requires_grad=False)
            w1_ms_quant_scale_param = np.full(shape=w1_ms_param.shape[-1], fill_value=w1_ms_quant_scale_param.item())
            self.parameter_dict[w1_ms_quant_scale_name] = ms.Parameter(
                ms.from_numpy(w1_ms_quant_scale_param).astype(ms.float16),
                name=w1_ms_name,
                requires_grad=False)
            w1_ms_quant_zp_param = np.full(shape=w1_ms_param.shape[-1], fill_value=w1_ms_quant_zp_param.item())
            self.parameter_dict[w1_ms_quant_zp_name] = ms.Parameter(ms.from_numpy(w1_ms_quant_zp_param).astype(ms.int8),
                                                                    name=w1_ms_name,
                                                                    requires_grad=False)
            self.parameter_dict[w1_ms_quant_bias_name] = ms.Parameter(
                ms.from_numpy(w1_ms_quant_bias_param).astype(ms.int32),
                name=w1_ms_name,
                requires_grad=False)
            w1_ms_scale_param = np.frombuffer(w1_ms_scale_param.astype(np.float32).tobytes(), dtype=np.int32).astype(
                np.int64)
            self.parameter_dict[w1_ms_scale_name] = ms.Parameter(ms.from_numpy(w1_ms_scale_param).astype(ms.int64),
                                                                 name=w1_ms_name,
                                                                 requires_grad=False)

            self.parameter_dict[w3_ms_name] = ms.Parameter(ms.from_numpy(w3_ms_param).astype(ms.int8),
                                                           name=w3_ms_name,
                                                           requires_grad=False)
            w3_ms_quant_scale_param = np.full(shape=w3_ms_param.shape[-1], fill_value=w3_ms_quant_scale_param.item())
            self.parameter_dict[w3_ms_quant_scale_name] = ms.Parameter(
                ms.from_numpy(w3_ms_quant_scale_param).astype(ms.float16),
                name=w1_ms_name,
                requires_grad=False)
            w3_ms_quant_zp_param = np.full(shape=w3_ms_param.shape[-1], fill_value=w3_ms_quant_zp_param.item())
            self.parameter_dict[w3_ms_quant_zp_name] = ms.Parameter(ms.from_numpy(w3_ms_quant_zp_param).astype(ms.int8),
                                                                    name=w1_ms_name,
                                                                    requires_grad=False)
            self.parameter_dict[w3_ms_quant_bias_name] = ms.Parameter(
                ms.from_numpy(w3_ms_quant_bias_param).astype(ms.int32),
                name=w1_ms_name,
                requires_grad=False)
            w3_ms_scale_param = np.frombuffer(w3_ms_scale_param.astype(np.float32).tobytes(), dtype=np.int32).astype(
                np.int64)
            self.parameter_dict[w3_ms_scale_name] = ms.Parameter(ms.from_numpy(w3_ms_scale_param).astype(ms.int64),
                                                                 name=w1_ms_name,
                                                                 requires_grad=False)

        self.parameter_dict[w2_ms_name] = ms.Parameter(ms.from_numpy(w2_ms_param).astype(ms.float16),
                                                       name=w2_ms_name,
                                                       requires_grad=False)

    def infer_process_norm_weight(self, src_hf_dir, layer_id, hf_weight_map):
        """infer process attention weight"""
        # attention_norm
        attention_norm_hf_name = f"model.layers.{layer_id}.input_layernorm.weight"
        attention_norm_ms_name = self.convert_weight_name(attention_norm_hf_name)
        attention_norm_ms_param, _ = self.get_safetensor_from_file(attention_norm_hf_name,
                                                                   src_hf_dir,
                                                                   hf_weight_map)
        att_norm_dtype = convert_np_to_ms_dtype(attention_norm_ms_param)
        self.parameter_dict[attention_norm_ms_name] = ms.Parameter(
            ms.from_numpy(attention_norm_ms_param).astype(att_norm_dtype),
            name=attention_norm_ms_name,
            requires_grad=False)

        # ffn_norm
        ffn_norm_hf_name = f"model.layers.{layer_id}.post_attention_layernorm.weight"
        ffn_norm_ms_name = self.convert_weight_name(ffn_norm_hf_name)
        ffn_norm_ms_param, _ = self.get_safetensor_from_file(ffn_norm_hf_name, src_hf_dir, hf_weight_map)
        ffn_norm_dtype = convert_np_to_ms_dtype(ffn_norm_ms_param)
        self.parameter_dict[ffn_norm_ms_name] = ms.Parameter(
            ms.from_numpy(ffn_norm_ms_param).astype(ffn_norm_dtype),
            name=ffn_norm_ms_name,
            requires_grad=False)

    def infer_convert_layer_weight(self, src_hf_dir, layer_id, hf_weight_map):
        """infer convert layer weight"""
        if self.is_quant:
            self.infer_process_quant_attention_weight(src_hf_dir, layer_id, hf_weight_map)
            self.infer_process_quant_ffn_weight(src_hf_dir, layer_id, hf_weight_map)
        else:
            self.infer_process_attention_weight(src_hf_dir, layer_id, hf_weight_map)
            self.infer_process_dense_ffn_weight(src_hf_dir, layer_id, hf_weight_map)
        self.infer_process_norm_weight(src_hf_dir, layer_id, hf_weight_map)

    def load_safetensors_shard(self, src_hf_dir):
        """qwen load safetensors and shard """
        rank_id = get_rank()
        param_json_path = ""
        for file in os.listdir(src_hf_dir):
            if file.endswith('index.json'):
                param_json_path = os.path.join(src_hf_dir, file)
                break

        hf_weight_map = {}
        if os.path.exists(param_json_path):
            with open(param_json_path, "r") as fp:
                hf_weight_map = json.load(fp)['weight_map']
        else:
            # only one safetensor, create a hf_weight_map
            safetensor_file = ""
            for file in os.listdir(src_hf_dir):
                if file.endswith('.safetensors'):
                    safetensor_file = file
                    break
            with safe_open(os.path.join(src_hf_dir, safetensor_file), framework="np") as sf_file:
                all_keys = sf_file.keys()
                for key in all_keys:
                    hf_weight_map[str(key).strip()] = safetensor_file

        quantization_config = self.config.model.model_config.quantization_config
        quant_method = quantization_config.quant_method if quantization_config else None

        self.infer_convert_outer_weight(src_hf_dir, hf_weight_map)
        num_layers = self.config.model.model_config.num_layers
        enable_tqdm = rank_id == 0
        start_layer, end_layer = self.get_layer_index(num_layers)
        for layer_id in tqdm(range(start_layer, end_layer), desc="Weight loading", disable=not enable_tqdm):
            self.infer_convert_layer_weight(src_hf_dir, layer_id, hf_weight_map)

        ms.load_param_into_net(self.network, self.parameter_dict)
        del self.parameter_dict
        gc.collect()
