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

from vllm_mindspore.model_executor.models.mf_models.weight_processor import BaseWeightProcessor, EPMethod
from vllm.logger import init_logger

logger = init_logger(__name__)

class Qwen2WeightProcessor(BaseWeightProcessor):
    r"""
    Provide Qwen2 Model weight load and shards.
    Args:
        config (Qwen2Config): The config of Qwen2 model.
        network (InferenceQwen2ForCausalLM): The network of Qwen2.

    """

    def __init__(self, config, network, is_quant):
        super().__init__(config, network, is_quant)

    def infer_convert_outer_weight(self, src_hf_dir, hf_weight_map):
        """convert weight not in model"""
        embed_tokens_hf_name = "model.embed_tokens.weight"
        embed_tokens_ms_name = self.convert_weight_name(embed_tokens_hf_name)
        if self.config.parallel_config.vocab_emb_dp:
            np_data, _ = self.get_safetensor_from_file(embed_tokens_hf_name, src_hf_dir, hf_weight_map)
        else:
            np_data, _ = self.get_safetensor_from_file_split_tp_group(embed_tokens_hf_name, src_hf_dir, hf_weight_map,
                                                                      split_axis=0)
        self.parameter_dict[embed_tokens_ms_name] = ms.Parameter(ms.from_numpy(np_data).astype(ms.bfloat16),
                                                                 name=embed_tokens_ms_name,
                                                                 requires_grad=False)

        norm_hf_name = "model.norm.weight"
        norm_ms_name = self.convert_weight_name(norm_hf_name)
        np_data, _ = self.get_safetensor_from_file(norm_hf_name, src_hf_dir, hf_weight_map)
        self.parameter_dict[norm_ms_name] = ms.Parameter(ms.from_numpy(np_data).astype(ms.bfloat16),
                                                         name=norm_ms_name,
                                                         requires_grad=False)

        lm_head_hf_name = "lm_head.weight"
        lm_head_ms_name = self.convert_weight_name(lm_head_hf_name)
        if not self.config.model.model_config.tie_word_embeddings:
            if not self.config.parallel_config.vocab_emb_dp:
                np_data, _ = self.get_safetensor_from_file_split_tp_group(lm_head_hf_name, src_hf_dir, hf_weight_map,
                                                                          split_axis=0)
            else:
                np_data, _ = self.get_safetensor_from_file(lm_head_hf_name, src_hf_dir, hf_weight_map)
            self.parameter_dict[lm_head_ms_name] = ms.Parameter(ms.from_numpy(np_data).astype(ms.bfloat16),
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

    def infer_process_dense_ffn_weight(self, src_hf_dir, layer_id, hf_weight_map):
        """infer process dense ffn weight"""

        ffn_concat = self.config.model.model_config.qkv_concat
        w1_hf_name = f"model.layers.{layer_id}.mlp.gate_proj.weight"
        w1_ms_name = self.convert_weight_name(w1_hf_name)
        w1_ms_param, _ = self.get_safetensor_from_file_split_tp_group(w1_hf_name, src_hf_dir, hf_weight_map,
                                                                      split_axis=0)

        w2_hf_name = f"model.layers.{layer_id}.mlp.down_proj.weight"
        w2_ms_name = self.convert_weight_name(w2_hf_name)
        w2_ms_param, _ = self.get_safetensor_from_file_split_tp_group(w2_hf_name, src_hf_dir, hf_weight_map,
                                                                      split_axis=1)

        w3_hf_name = f"model.layers.{layer_id}.mlp.up_proj.weight"
        w3_ms_name = self.convert_weight_name(w3_hf_name)
        w3_ms_param, _ = self.get_safetensor_from_file_split_tp_group(w3_hf_name, src_hf_dir, hf_weight_map,
                                                                      split_axis=0)

        if ffn_concat:
            w_gate_hidden_name = f"model.layers.{layer_id}.feed_forward.w_gate_hidden.weight"
            w_gate_hidden_param = np.concatenate((w1_ms_param, w3_ms_param), axis=0)
            self.parameter_dict[w_gate_hidden_name] = ms.Parameter(w_gate_hidden_param, name=w_gate_hidden_name,
                                                                   requires_grad=False)
        else:
            self.parameter_dict[w1_ms_name] = ms.Parameter(ms.from_numpy(w1_ms_param).astype(ms.bfloat16),
                                                           name=w1_ms_name,
                                                           requires_grad=False)
            self.parameter_dict[w3_ms_name] = ms.Parameter(ms.from_numpy(w3_ms_param).astype(ms.bfloat16),
                                                           name=w3_ms_name,
                                                           requires_grad=False)

        self.parameter_dict[w2_ms_name] = ms.Parameter(ms.from_numpy(w2_ms_param).astype(ms.bfloat16),
                                                       name=w2_ms_name,
                                                       requires_grad=False)

    def infer_process_attention_weight(self, src_hf_dir, layer_id, hf_weight_map):
        """infer process attention weight"""
        qkv_concat = self.config.model.model_config.qkv_concat
        # wq
        wq_hf_name = f"model.layers.{layer_id}.self_attn.q_proj.weight"
        wq_ms_name = self.convert_weight_name(wq_hf_name)
        wq_ms_param, _ = self.get_safetensor_from_file_split_tp_group(wq_hf_name, src_hf_dir, hf_weight_map,
                                                                      split_axis=0)
        # wq bias
        wq_bias_hf_name = f"model.layers.{layer_id}.self_attn.q_proj.bias"
        wq_bias_ms_name = self.convert_weight_name(wq_bias_hf_name)
        wq_bias_ms_param, _ = self.get_safetensor_from_file_split_tp_group(wq_bias_hf_name, src_hf_dir, hf_weight_map,
                                                                           split_axis=0)

        # wk
        wk_hf_name = f"model.layers.{layer_id}.self_attn.k_proj.weight"
        wk_ms_name = self.convert_weight_name(wk_hf_name)
        wk_ms_param, _ = self.get_safetensor_from_file_split_tp_group(wk_hf_name, src_hf_dir, hf_weight_map,
                                                                      split_axis=0)
        # wk bias
        wk_bias_hf_name = f"model.layers.{layer_id}.self_attn.k_proj.bias"
        wk_bias_ms_name = self.convert_weight_name(wk_bias_hf_name)
        wk_bias_ms_param, _ = self.get_safetensor_from_file_split_tp_group(wk_bias_hf_name, src_hf_dir, hf_weight_map,
                                                                           split_axis=0)

        # wv
        wv_hf_name = f"model.layers.{layer_id}.self_attn.v_proj.weight"
        wv_ms_name = self.convert_weight_name(wv_hf_name)
        wv_ms_param, _ = self.get_safetensor_from_file_split_tp_group(wv_hf_name, src_hf_dir, hf_weight_map,
                                                                      split_axis=0)
        # wv bias
        wv_bias_hf_name = f"model.layers.{layer_id}.self_attn.v_proj.bias"
        wv_bias_ms_name = self.convert_weight_name(wv_bias_hf_name)
        wv_bias_ms_param, _ = self.get_safetensor_from_file_split_tp_group(wv_bias_hf_name, src_hf_dir, hf_weight_map,
                                                                           split_axis=0)

        if qkv_concat:
            w_qkv_name = f"model.layers.{layer_id}.attention.w_qkv.weight"
            w_qkv_param = np.concatenate((wq_ms_param, wk_ms_param, wv_ms_param), axis=0)
            w_qkv_param = ms.from_numpy(w_qkv_param).astype(ms.bfloat16)
            self.parameter_dict[w_qkv_name] = ms.Parameter(w_qkv_param, name=w_qkv_name, requires_grad=False)

            w_qkv_bias_name = f"model.layers.{layer_id}.attention.w_qkv.bias"
            w_qkv_bias_param = np.concatenate((wq_bias_ms_param, wk_bias_ms_param, wv_bias_ms_param), axis=0)
            w_qkv_bias_param = ms.from_numpy(w_qkv_bias_param).astype(ms.bfloat16)
            self.parameter_dict[w_qkv_bias_name] = ms.Parameter(w_qkv_bias_param, name=w_qkv_bias_name,
                                                                requires_grad=False)
        else:
            self.parameter_dict[wq_ms_name] = ms.Parameter(ms.from_numpy(wq_ms_param).astype(ms.bfloat16),
                                                           name=wq_ms_name,
                                                           requires_grad=False)
            self.parameter_dict[wk_ms_name] = ms.Parameter(ms.from_numpy(wk_ms_param).astype(ms.bfloat16),
                                                           name=wk_ms_name,
                                                           requires_grad=False)
            self.parameter_dict[wv_ms_name] = ms.Parameter(ms.from_numpy(wv_ms_param).astype(ms.bfloat16),
                                                           name=wv_ms_name,
                                                           requires_grad=False)

            self.parameter_dict[wq_bias_ms_name] = ms.Parameter(
                ms.from_numpy(wq_bias_ms_param).astype(ms.bfloat16),
                name=wq_bias_ms_name,
                requires_grad=False)
            self.parameter_dict[wk_bias_ms_name] = ms.Parameter(
                ms.from_numpy(wk_bias_ms_param).astype(ms.bfloat16),
                name=wk_bias_ms_name,
                requires_grad=False)
            self.parameter_dict[wv_bias_ms_name] = ms.Parameter(
                ms.from_numpy(wv_bias_ms_param).astype(ms.bfloat16),
                name=wv_bias_ms_name,
                requires_grad=False)

        # wo
        wo_hf_name = f"model.layers.{layer_id}.self_attn.o_proj.weight"
        wo_ms_name = self.convert_weight_name(wo_hf_name)
        wo_ms_param, _ = self.get_safetensor_from_file_split_tp_group(wo_hf_name, src_hf_dir, hf_weight_map,
                                                                      split_axis=1)
        self.parameter_dict[wo_ms_name] = ms.Parameter(ms.from_numpy(wo_ms_param).astype(ms.bfloat16),
                                                       name=wo_ms_name,
                                                       requires_grad=False)

    def infer_process_norm_weight(self, src_hf_dir, layer_id, hf_weight_map):
        """infer process attention weight"""
        # attention_norm
        attention_norm_hf_name = f"model.layers.{layer_id}.input_layernorm.weight"
        attention_norm_ms_name = self.convert_weight_name(attention_norm_hf_name)
        attention_norm_ms_param, _ = self.get_safetensor_from_file(attention_norm_hf_name,
                                                                   src_hf_dir,
                                                                   hf_weight_map)
        self.parameter_dict[attention_norm_ms_name] = ms.Parameter(
            ms.from_numpy(attention_norm_ms_param).astype(ms.bfloat16),
            name=attention_norm_ms_name,
            requires_grad=False)

        # ffn_norm
        ffn_norm_hf_name = f"model.layers.{layer_id}.post_attention_layernorm.weight"
        ffn_norm_ms_name = self.convert_weight_name(ffn_norm_hf_name)
        ffn_norm_ms_param, _ = self.get_safetensor_from_file(ffn_norm_hf_name, src_hf_dir, hf_weight_map)
        self.parameter_dict[ffn_norm_ms_name] = ms.Parameter(
            ms.from_numpy(ffn_norm_ms_param).astype(ms.bfloat16),
            name=ffn_norm_ms_name,
            requires_grad=False)

    def infer_convert_layer_weight(self, src_hf_dir, layer_id, hf_weight_map):
        """infer convert layer weight"""
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
            safetensor_file = "model.safetensors"
            with safe_open(f"{src_hf_dir}/{safetensor_file}", framework="np") as sf_file:
                all_keys = sf_file.keys()
                for key in all_keys:
                    hf_weight_map[str(key).strip()] = safetensor_file

        self.infer_convert_outer_weight(src_hf_dir, hf_weight_map)
        num_layers = self.config.model.model_config.num_layers
        enable_tqdm = rank_id == 0
        for layer_id in tqdm(range(num_layers), desc="Weight loading", disable=not enable_tqdm):
            self.infer_convert_layer_weight(src_hf_dir, layer_id, hf_weight_map)

        param_not_load, ckpt_not_load = ms.load_param_into_net(self.network, self.parameter_dict)
        logger.info("param_not_load: %s, ckpt_not_load: %s" % (str(param_not_load), str(ckpt_not_load)))
        del self.parameter_dict
        gc.collect()
