# SPDX-License-Identifier: Apache-2.0

# Adapted from
# https://github.com/vllm-project/vllm/blob/v0.8.3/vllm/model_executor/model_loader/weight_utils.py
#
# Copyright 2025 Huawei Technologies Co., Ltd.
# Copyright 2024-2025 The vLLM team.
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
import glob
import json
import os
from collections.abc import Generator
from typing import Any

import huggingface_hub
import mindspore as ms
import numpy as np
from huggingface_hub import snapshot_download
from mindspore import Parameter
from safetensors import safe_open
from tqdm.auto import tqdm
from vllm.config import LoadConfig
from vllm.model_executor.model_loader.weight_utils import (DisabledTqdm,
                                                           get_lock)

from vllm_mindspore.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm_mindspore.platforms.ascend import ModelConfig
from vllm_mindspore.utils import is_310p
from vllm.model_executor.model_loader.weight_utils import (_BAR_FORMAT,
                                                           enable_tqdm)


def split_loaded_weight(loaded_weight, shard_dim, start_idx, shard_size):
    """
    Read numpy slice data based on axis and slice range.
    :loaded_weight: PySafeSlice object
    :shard_dim: axis of weight slice
    :start_idx: start slice index
    :shard_size: end slice index
    """
    if shard_dim is None:
        loaded_weight = loaded_weight[:]
        return loaded_weight

    end_idx = start_idx + shard_size
    if shard_dim == 0:
        loaded_weight = loaded_weight[start_idx:end_idx]
    elif shard_dim == 1:
        loaded_weight = loaded_weight[:, start_idx:end_idx]
    elif shard_dim == 2:
        loaded_weight = loaded_weight[:, :, start_idx:end_idx]
    else:
        raise ValueError("shard_dim:{} is not supported.".format(shard_dim))
    return loaded_weight


def safetensors_weights_iterator(
    hf_weights_files: list[str],
    use_tqdm_on_load: bool,
) -> Generator[tuple[str, Any], None, None]:
    """Iterate over the weights in the model safetensor files."""
    for st_file in tqdm(
            hf_weights_files,
            desc="Loading safetensors checkpoint shards",
            disable=not enable_tqdm(use_tqdm_on_load),
            bar_format=_BAR_FORMAT,
    ):
        with safe_open(st_file, framework="np") as f:
            for name in f.keys():  # noqa: SIM118
                # TODO： use slice
                x = f.get_tensor(name)
                x = x.astype(np.float16) \
                    if (str(x.dtype) == 'bfloat16' and is_310p()) else x
                yield name, ms.tensor(x)


def default_weight_loader(param: Parameter, loaded_weight: Any) -> None:
    """Default weight loader."""
    loaded_weight = loaded_weight[:]
    param.set_data(ms.Tensor(loaded_weight, dtype=param.dtype))


def get_quant_config(model_config: ModelConfig,
                     load_config: LoadConfig) -> QuantizationConfig:

    from vllm_mindspore.model_executor.layers.quantization import (
        get_quantization_config)
    quant_cls = get_quantization_config(model_config.quantization)

    # GGUF doesn't have config file
    if model_config.quantization == "gguf":
        return quant_cls.from_config({})

    # Read the quantization config from the HF model config, if available.
    hf_quant_config = getattr(model_config.hf_config, "quantization_config",
                              None)
    # some vision model may keep quantization_config in their text_config
    hf_text_config = getattr(model_config.hf_config, "text_config", None)
    if hf_quant_config is None and hf_text_config is not None:
        hf_quant_config = getattr(hf_text_config, "quantization_config", None)
    if hf_quant_config is None:
        # compressed-tensors uses a compressions_config
        hf_quant_config = getattr(model_config.hf_config, "compression_config",
                                  None)
    if hf_quant_config is not None:
        if os.path.isdir(model_config.model):
            quant_config_file = os.path.join(
                model_config.model,
                quant_cls.get_config_filenames()[0])
            with open(quant_config_file) as f:
                quant_config = json.load(f)
        return quant_cls.from_config(hf_quant_config | quant_config)

    # In case of bitsandbytes/QLoRA, get quant config from the adapter model.
    if model_config.quantization == "bitsandbytes":
        if (not load_config.model_loader_extra_config
                or "qlora_adapter_name_or_path"
                not in load_config.model_loader_extra_config):
            return quant_cls.from_config({"adapter_name_or_path": ""})
        model_name_or_path = load_config.model_loader_extra_config[
            "qlora_adapter_name_or_path"]

    else:
        model_name_or_path = model_config.model
    is_local = os.path.isdir(model_name_or_path)
    if not is_local:
        # Download the config files.
        with get_lock(model_name_or_path, load_config.download_dir):
            hf_folder = snapshot_download(
                model_name_or_path,
                revision=model_config.revision,
                allow_patterns="*.json",
                cache_dir=load_config.download_dir,
                local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE,
                tqdm_class=DisabledTqdm,
            )
    else:
        hf_folder = model_name_or_path

    possible_config_filenames = quant_cls.get_config_filenames()

    # If the quantization config is not found, use the default config.
    if not possible_config_filenames:
        return quant_cls()

    config_files = glob.glob(os.path.join(hf_folder, "*.json"))

    quant_config_files = [
        f for f in config_files if any(
            f.endswith(x) for x in possible_config_filenames)
    ]
    if len(quant_config_files) == 0:
        raise ValueError(
            f"Cannot find the config file for {model_config.quantization}")
    if len(quant_config_files) > 1:
        raise ValueError(
            f"Found multiple config files for {model_config.quantization}: "
            f"{quant_config_files}")

    quant_config_file = quant_config_files[0]
    with open(quant_config_file) as f:
        config = json.load(f)

        if model_config.quantization == "bitsandbytes":
            config["adapter_name_or_path"] = model_name_or_path
        elif model_config.quantization == "modelopt":
            if config["producer"]["name"] == "modelopt":
                return quant_cls.from_config(config)
            else:
                raise ValueError(
                    f"Unsupported quantization config"
                    f" found for {model_config.quantization} in {f}.")

    return quant_cls.from_config(config)
