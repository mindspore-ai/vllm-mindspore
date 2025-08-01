# SPDX-License-Identifier: Apache-2.0

# Adapted from
# https://github.com/vllm-project/vllm/blob/v0.9.1/vllm/model_executor/model_loader/default_loader.py
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
import os
from typing import Optional

from transformers.utils import SAFE_WEIGHTS_INDEX_NAME
from vllm.config import LoadFormat
from vllm.distributed import get_tensor_model_parallel_rank
from vllm.model_executor.model_loader.weight_utils import (
    download_safetensors_index_file_from_hf, download_weights_from_hf,
    filter_duplicate_safetensors_files, filter_files_not_needed_for_inference)


def _prepare_weights(
    self,
    model_name_or_path: str,
    revision: Optional[str],
    fall_back_to_pt: bool,
    allow_patterns_overrides: Optional[list[str]],
) -> tuple[str, list[str], bool]:
    """Prepare weights for the model.

        If the model is not local, it will be downloaded."""
    model_name_or_path = (self._maybe_download_from_modelscope(
        model_name_or_path, revision) or model_name_or_path)

    is_local = os.path.isdir(model_name_or_path)
    load_format = self.load_config.load_format
    use_safetensors = False
    index_file = SAFE_WEIGHTS_INDEX_NAME
    # Some quantized models use .pt files for storing the weights.
    if load_format == LoadFormat.AUTO:
        allow_patterns = ["*.safetensors", "*.bin"]
    elif (load_format == LoadFormat.SAFETENSORS
          or load_format == LoadFormat.FASTSAFETENSORS):
        use_safetensors = True
        allow_patterns = ["*.safetensors"]
    elif load_format == LoadFormat.MISTRAL:
        use_safetensors = True
        allow_patterns = ["consolidated*.safetensors"]
        index_file = "consolidated.safetensors.index.json"
    elif load_format == LoadFormat.PT:
        allow_patterns = ["*.pt"]
    elif load_format == LoadFormat.NPCACHE:
        allow_patterns = ["*.bin"]
    else:
        raise ValueError(f"Unknown load_format: {load_format}")

    if fall_back_to_pt:
        allow_patterns += ["*.pt"]

    if allow_patterns_overrides is not None:
        allow_patterns = allow_patterns_overrides

    if not is_local:
        hf_folder = download_weights_from_hf(
            model_name_or_path,
            self.load_config.download_dir,
            allow_patterns,
            revision,
            ignore_patterns=self.load_config.ignore_patterns,
        )
    else:
        hf_folder = model_name_or_path
    hf_weights_files: list[str] = []
    for pattern in allow_patterns:
        hf_weights_files += glob.glob(os.path.join(hf_folder, pattern))
        if len(hf_weights_files) == 0:
            tp_rank = get_tensor_model_parallel_rank()
            hf_weights_files += glob.glob(
                os.path.join(hf_folder, f"rank_{tp_rank}", pattern))
        if len(hf_weights_files) > 0:
            if pattern == "*.safetensors":
                use_safetensors = True
            break
    if use_safetensors:
        # For models like Mistral-7B-Instruct-v0.3
        # there are both sharded safetensors files and a consolidated
        # safetensors file. Using both breaks.
        # Here, we download the `model.safetensors.index.json` and filter
        # any files not found in the index.
        if not is_local:
            download_safetensors_index_file_from_hf(
                model_name_or_path,
                index_file,
                self.load_config.download_dir,
                revision,
            )
        hf_weights_files = filter_duplicate_safetensors_files(
            hf_weights_files, hf_folder, index_file)
    else:
        hf_weights_files = filter_files_not_needed_for_inference(
            hf_weights_files)

    if len(hf_weights_files) == 0:
        raise RuntimeError(
            f"Cannot find any model weights with `{model_name_or_path}`")

    return hf_folder, hf_weights_files, use_safetensors
