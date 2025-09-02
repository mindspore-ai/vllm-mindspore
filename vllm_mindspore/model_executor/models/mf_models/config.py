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
import types

import numpy as np
from mindformers.tools.register.config import MindFormerConfig
from vllm import envs
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group

from vllm_mindspore.utils import is_310p

MF_CTX_MAPPING = {
    'run_mode': (None, "predict"),
    'use_legacy': (None, False),
    'load_ckpt_format': (None, 'safetensors'),
    'auto_trans_ckpt': (None, True),
}

# yapf: disable # noqa: ERA001
MF_PARALLEL_MAPPING = {
    'parallel_mode': (None, 'STAND_ALONE'),
    'parallel_config.model_parallel': ('parallel_config.tensor_parallel_size', None),  # noqa: E501
    'parallel_config.pipeline_stage': ('parallel_config.pipeline_parallel_size', None),  # noqa: E501
    'parallel_config.vocab_emb_dp': (None, False)
}
# yapf: enable # noqa: ERA001

# model default config
MODEL_RELATED_MAPPING = {
    'qwen3': {
        'params_dtype': 'bfloat16',  # need an input
    },
    'qwen3_moe': {
        'params_dtype': 'bfloat16',  # need an input
        'layernorm_compute_dtype': 'bfloat16',
        'rotary_dtype': 'bfloat16',
        'router_dense_type': 'bfloat16',
    }
    # Add anther model type...
}

MODEL_RELATED_MAPPING_310p = {
    'qwen3': {
        'params_dtype': 'float16',  # need an input
        'compute_dtype': 'float16',
    },
    # Add anther model type...
}


def get_nested_attr(obj, path: str, default=None):
    """get nested attr from obj."""
    current = obj
    for attr in path.split('.'):
        if not hasattr(current, attr):
            return default
        current = getattr(current, attr)
    return current


def set_nested_attr(obj, path: str, value):
    """Set nested attr of MindFormerConfig."""
    attrs = path.split('.')

    current = obj
    for attr in attrs[:-1]:
        if not hasattr(current, attr) or getattr(current, attr) is None:
            setattr(current, attr, MindFormerConfig())
        current = getattr(current, attr)

    setattr(current, attrs[-1], value)


def transform_config(mapping_table: dict, vllm_config: VllmConfig,
                     target_config):
    for target_path, mapping in mapping_table.items():
        src_path, transform = mapping

        src_value = get_nested_attr(vllm_config,
                                    src_path) if src_path is not None else None

        if src_value is not None:
            transformed_value = src_value
        elif transform and isinstance(
                transform, (types.FunctionType, types.BuiltinFunctionType)):
            transformed_value = transform(src_value)
        else:
            transformed_value = transform

        if transformed_value is not None:
            set_nested_attr(target_config, target_path, transformed_value)


def gen_model_relatived_config(model_type):
    if is_310p():
        return MODEL_RELATED_MAPPING_310p.get(model_type)
    return MODEL_RELATED_MAPPING.get(model_type)


def get_mf_offset(vllm_config: VllmConfig):
    """ get pp offset from vllm style"""
    partition_list_str = envs.VLLM_PP_LAYER_PARTITION
    num_layers = vllm_config.model_config.hf_config.num_hidden_layers
    pp_size = vllm_config.parallel_config.pipeline_parallel_size
    if partition_list_str is not None:
        try:
            partitions = [
                int(layer) for layer in partition_list_str.split(",")
            ]
        except ValueError as err:
            raise ValueError("Invalid partition string: {}".format(
                partition_list_str)) from err
        if len(partitions) != pp_size:
            raise ValueError(f"{len(partitions)=} does not match {pp_size=}.")
        if sum(partitions) != num_layers:
            raise ValueError(
                f"{sum(partitions)=} does not match {num_layers=}.")
        partitions = np.array(partitions, dtype=np.int32)
        avg_layers = num_layers // pp_size
        avg_layers_list = np.ones((pp_size, ), dtype=np.int32) * avg_layers
        if (partitions == avg_layers_list).all():
            return 0
        else:
            return (partitions - avg_layers_list).tolist()
    else:
        return 0


def gen_model_config_dict(vllm_config: VllmConfig):
    target_config = MindFormerConfig()

    model_type = vllm_config.model_config.hf_config.model_type
    model_related_config = gen_model_relatived_config(model_type)
    target_config.update(model_related_config)
    target_config.pre_process = get_pp_group().is_first_rank
    target_config.post_process = get_pp_group().is_last_rank
    target_config.offset = get_mf_offset(vllm_config)

    return target_config


def gen_mf_config(vllm_config: VllmConfig):
    target_config = MindFormerConfig()
    transform_config(MF_CTX_MAPPING, vllm_config, target_config)
    transform_config(MF_PARALLEL_MAPPING, vllm_config, target_config)
    target_config.set_value(
        'model.model_config',
        MindFormerConfig(**gen_model_config_dict(vllm_config)))
    return target_config
