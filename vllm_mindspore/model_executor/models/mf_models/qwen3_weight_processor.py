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
import numpy as np

import mindspore as ms

from vllm_mindspore.model_executor.models.mf_models.qwen2_weight_processor import Qwen2WeightProcessor


class Qwen3WeightProcessor(Qwen2WeightProcessor):
    r"""
    Provide Qwen3 Model weight load and shards.
    Args:
        config (Qwen3Config): The config of Qwen3 model.
        network (InferenceQwen3ForCausalLM): The network of Qwen3.

    """

    def __init__(self, config, network, is_quant):
        super().__init__(config, network, is_quant)

    def convert_weight_name(self, weight_name: str):
        """replace weight name"""
        weight_name = weight_name.replace('embed_tokens.weight', 'tok_embeddings.embedding_weight')
        weight_name = weight_name.replace('self_attn.q_proj.', 'attention.wq.')
        weight_name = weight_name.replace('self_attn.k_proj.', 'attention.wk.')
        weight_name = weight_name.replace('self_attn.v_proj.', 'attention.wv.')
        weight_name = weight_name.replace('self_attn.o_proj.', 'attention.wo.')
        weight_name = weight_name.replace('self_attn.q_norm.', 'attention.q_norm.')
        weight_name = weight_name.replace('self_attn.k_norm.', 'attention.k_norm.')

        weight_name = weight_name.replace('mlp.gate_proj.', 'feed_forward.w1.')
        weight_name = weight_name.replace('mlp.down_proj.', 'feed_forward.w2.')
        weight_name = weight_name.replace('mlp.up_proj.', 'feed_forward.w3.')
        weight_name = weight_name.replace('.input_layernorm.', '.attention_norm.')
        weight_name = weight_name.replace('.post_attention_layernorm.', '.ffn_norm.')
        weight_name = weight_name.replace('model.norm.weight', 'model.norm_out.weight')
        return weight_name

    def infer_process_attention_weight(self, src_hf_dir, layer_id, hf_weight_map):
        """infer process attention weight"""
        qkv_concat = self.config.model.model_config.qkv_concat
        # wq
        wq_hf_name = f"model.layers.{layer_id}.self_attn.q_proj.weight"
        wq_ms_name = self.convert_weight_name(wq_hf_name)
        wq_ms_param, _ = self.get_safetensor_from_file(wq_hf_name, src_hf_dir, hf_weight_map, is_split_param=True,
                                                       split_axis=0)

        # wk
        wk_hf_name = f"model.layers.{layer_id}.self_attn.k_proj.weight"
        wk_ms_name = self.convert_weight_name(wk_hf_name)
        wk_ms_param, _ = self.get_safetensor_from_file(wk_hf_name, src_hf_dir, hf_weight_map, is_split_param=True,
                                                       split_axis=0)

        # wv
        wv_hf_name = f"model.layers.{layer_id}.self_attn.v_proj.weight"
        wv_ms_name = self.convert_weight_name(wv_hf_name)
        wv_ms_param, _ = self.get_safetensor_from_file(wv_hf_name, src_hf_dir, hf_weight_map, is_split_param=True,
                                                       split_axis=0)

        # wq_norm
        q_norm_hf_name = f"model.layers.{layer_id}.self_attn.q_norm.weight"
        q_norm_ms_name = self.convert_weight_name(q_norm_hf_name)
        q_norm_ms_param, _ = self.get_safetensor_from_file(q_norm_hf_name, src_hf_dir, hf_weight_map)
        self.parameter_dict[q_norm_ms_name] = ms.Parameter(ms.Tensor(q_norm_ms_param, ms.bfloat16), name=q_norm_ms_name,
                                                           requires_grad=False)

        #wk_norm
        k_norm_hf_name = f"model.layers.{layer_id}.self_attn.k_norm.weight"
        k_norm_ms_name = self.convert_weight_name(k_norm_hf_name)
        k_norm_ms_param, _ = self.get_safetensor_from_file(k_norm_hf_name, src_hf_dir, hf_weight_map)
        self.parameter_dict[k_norm_ms_name] = ms.Parameter(ms.Tensor(k_norm_ms_param, ms.bfloat16), name=k_norm_ms_name,
                                                           requires_grad=False)

        if qkv_concat:
            w_qkv_name = f"model.layers.{layer_id}.attention.w_qkv.weight"
            w_qkv_param = np.concatenate((wq_ms_param, wk_ms_param, wv_ms_param), axis=0)
            w_qkv_param = ms.from_numpy(w_qkv_param).astype(ms.bfloat16)
            self.parameter_dict[w_qkv_name] = ms.Parameter(w_qkv_param, name=w_qkv_name, requires_grad=False)

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

        # wo
        wo_hf_name = f"model.layers.{layer_id}.self_attn.o_proj.weight"
        wo_ms_name = self.convert_weight_name(wo_hf_name)
        wo_ms_param, _ = self.get_safetensor_from_file(wo_hf_name, src_hf_dir, hf_weight_map, is_split_param=True,
                                                       split_axis=1)
        self.parameter_dict[wo_ms_name] = ms.Parameter(ms.from_numpy(wo_ms_param).astype(ms.bfloat16),
                                                       name=wo_ms_name,
                                                       requires_grad=False)
