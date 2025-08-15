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
from functools import partial

from mindformers.tools.register.config import MindFormerConfig
from vllm.config import VllmConfig
from vllm.logger import init_logger

from vllm_mindspore.utils import is_310p

logger = init_logger(__name__)


def _get_addtional_conf(src_value, path, default=None):
    """
    Retrieve mindformers configuration from cli additional-config arguments.

    Args:
        src_value: Source data, expected to be a dictionary or None.
        key: The configuration key to retrieve.
        default: Default value to return if key is not found.

    Returns:
        The value corresponding to the specified key, or default if not found.

    Example:
        Example of vLLM MindSpore startup command:
        vllm-mindspore serve --model=/path/to/model --additional-config 
        '{"expert_parallel": 1, "mindformers": {"rotary_dtype": "float32"}}'

        >>> additional_config = vllm_config.additional_config
        >>> _get_addtional_conf(additional_config, 'expert_parallel')
        >>> 1
        >>> _get_addtional_conf(additional_config, 'mindformers.rotary_dtype')
        >>> float32
        >>> _get_addtional_conf(additional_config, 'mindformers.params_dtype', 'bfloat16')
        >>> bfloat16
    """
    if not isinstance(src_value, dict) or not path:
        return default

    current = src_value
    for key in path.split("."):
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


"""
The formatting of `MF_CTX_MAPPING` `MF_PARALLEL_MAPPING` `MF_COMMON_MAPPING`
mapping = {target_path, (source_path, value_or_function)}
target_path: Specifies the path to a configuration parameter within the nested
             structure of mindformers' YAML config file,
             using dot notation (".") to denote hierarchy levels.
source_path: Specifies the path to a configuration parameter in VllmConfig,
             with hierarchical levels delimited by periods ("."). 
             If the source_path is None, value_or_function will be used directly.
value_or_function: Specifies the default value for the configuration parameter
                   or a partial function for computing configuration values.
"""
# yapf: disable # noqa: ERA001
# flake8: noqa: E501
MF_CTX_MAPPING = {
    'run_mode': (None, "predict"),
    'use_legacy': (None, False),
    'load_ckpt_format': (None, 'safetensors'),
    'auto_trans_ckpt': (None, True),
}

MF_PARALLEL_MAPPING = {
    'parallel_mode': (None, 'STAND_ALONE'),
    'parallel_config.model_parallel': ('parallel_config.tensor_parallel_size', None),
    'parallel_config.pipeline_stage': ('parallel_config.pipeline_parallel_size', None),
    'parallel_config.data_parallel': ('parallel_config.data_parallel_size', None),
    'parallel_config.vocab_emb_dp': (None, False),
    'parallel_config.expert_parallel': ('additional_config', partial(_get_addtional_conf, path='expert_parallel'))
}

MF_MODEL_COMMON_MAPPING = {
    'model.model_config.compute_dtype': ('model_config.hf_config.torch_dtype', 'bfloat16'),
    'model.model_config.layernorm_compute_dtype': (None, 'bfloat16'),
    'model.model_config.rotary_dtype': (None, 'bfloat16'),
    'model.model_config.params_dtype': (None, 'bfloat16'),
    'model.model_config.router_dense_type': (None, 'bfloat16'),
}

MF_MODEL_COMMON_MAPPING_310p = {
    'model.model_config.compute_dtype': ('model_config.hf_config.torch_dtype', 'float16'),
    'model.model_config.layernorm_compute_dtype': (None, 'float16'),
    'model.model_config.rotary_dtype': (None, 'float16'),
    'model.model_config.params_dtype': (None, 'float16'),
    'model.model_config.router_dense_type': (None, 'float16'),
}
# yapf: enable # noqa: ERA001

# model default config
MODEL_RELATED_MAPPING = {
    # Add anther model type...
}


def _get_nested_attr(obj, path: str, default=None):
    """get nested attr from obj."""
    current = obj
    for attr in path.split('.'):
        if not hasattr(current, attr):
            return default
        current = getattr(current, attr)
    return current


def _set_nested_attr(obj, path: str, value):
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
    """
    Transform source configuration to target configuration format based on 
    mapping table.

    This function iterates through each target path in the mapping table, 
    retrieves the corresponding value from the source configuration, 
    applies transformation rules, and sets the result to the appropriate 
    location in the target configuration.

    Args:
        mapping_table (dict): Mapping table where keys are target configuration
                              paths and values are tuples of (source path,
                              default value or transformation function).
        vllm_config (VllmConfig): Source configuration object.
        target_config: Target configuration object.

    Returns:
        None, modifies target_config object directly.
    """
    for target_path, mapping in mapping_table.items():
        src_path, transform = mapping

        src_value = _get_nested_attr(
            vllm_config, src_path) if src_path is not None else None

        if src_value is not None and not callable(transform):
            transformed_value = src_value
        elif callable(transform):
            transformed_value = transform(src_value)
        else:
            transformed_value = transform

        if transformed_value is not None:
            _set_nested_attr(target_config, target_path, transformed_value)


def gen_model_config_dict(vllm_config: VllmConfig):
    """
    Generate model configuration dictionary based on MODEL_RELATED_MAPPING.
    """
    target_config = MindFormerConfig()

    model_type = vllm_config.model_config.hf_config.model_type
    model_related_config = MODEL_RELATED_MAPPING.get(model_type)
    if model_related_config is not None:
        target_config.update(model_related_config)
    target_config.post_process = False

    return target_config


def _merge_dicts(original_config, update_config):
    """
    Recursively update the original configuration dictionary with values from
    update_config.
    
    This function merges the update_config into original_config, with special
    handling for nested dictionaries. When both the original and update values
    at a key are dictionaries, it recursively merges them. Otherwise,
    it overwrites the original value with the update value.
    
    Args:
        original_config (dict): The original configuration dictionary to be updated.
        update_config (dict): The configuration dictionary containing updates.
        
    Returns:
        None. The original_config is modified in-place.
    """
    for key, value in update_config.items():
        if (key in original_config and isinstance(original_config[key], dict)
                and isinstance(value, dict)):
            _merge_dicts(original_config[key], value)
        else:
            original_config[key] = value


def gen_mf_config(vllm_config: VllmConfig):
    """Generate mindformers configuration."""
    target_config = MindFormerConfig()
    transform_config(MF_CTX_MAPPING, vllm_config, target_config)
    transform_config(MF_PARALLEL_MAPPING, vllm_config, target_config)
    target_config.set_value(
        'model.model_config',
        MindFormerConfig(**gen_model_config_dict(vllm_config)))
    if is_310p():
        transform_config(MF_MODEL_COMMON_MAPPING_310p, vllm_config,
                         target_config)
    else:
        transform_config(MF_MODEL_COMMON_MAPPING, vllm_config, target_config)
    # Update target config with additional config.
    # The configuration hierarchy in the additional config must match the
    # hierarchy structure of the MindFormers YAML configuration file.
    _merge_dicts(target_config,
                 vllm_config.additional_config.get('mindformers', {}))
    logger.info('The generated MindFormers config: %s', target_config)
    return target_config
