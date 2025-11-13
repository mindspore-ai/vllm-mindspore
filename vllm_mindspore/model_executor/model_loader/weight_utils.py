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

from collections.abc import Generator
from typing import Any

import mindspore as ms
from mindspore import Parameter
from safetensors import safe_open
from tqdm.auto import tqdm
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
    safetensors_load_strategy: str = "lazy",
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
                # Return a lightweight PySafeSlice object that uses file
                # pointer offset internally to read Safetensor on demand,
                # avoiding memory explosion. Actual data can be obtained
                # through slicing operation like param[start:end]
                param = f.get_slice(name)
                yield name, param


def default_weight_loader(param: Parameter, loaded_weight: Any) -> None:
    """Default weight loader."""
    loaded_weight = loaded_weight[:]
    param.set_data(ms.Tensor(loaded_weight, dtype=param.dtype))
