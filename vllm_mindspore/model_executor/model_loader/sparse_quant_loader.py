# SPDX-License-Identifier: Apache-2.0
#
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
"""Model loader for sparse quantization (W8A8SC) models."""

import glob
import os
from typing import Optional

from transformers.utils import SAFE_WEIGHTS_INDEX_NAME
from vllm.distributed import get_tensor_model_parallel_rank
from vllm.model_executor.model_loader import register_model_loader
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.model_loader.weight_utils import (
    download_safetensors_index_file_from_hf, download_weights_from_hf,
    filter_duplicate_safetensors_files, filter_files_not_needed_for_inference,
    maybe_download_from_modelscope)


def _find_rank_dir(hf_folder: str, tp_rank: Optional[int]) -> Optional[str]:
    """Find the appropriate rank directory for weight loading."""
    if not os.path.isdir(hf_folder):
        return None
    try:
        rank_dirs = [
            item for item in os.listdir(hf_folder)
            if os.path.isdir(os.path.join(hf_folder, item))
            and item.startswith("rank_")
        ]
        if not rank_dirs:
            return None
        # Prefer current rank, then rank_0, then first found
        for candidate in [
                f"rank_{tp_rank}" if tp_rank is not None else None, "rank_0",
                rank_dirs[0]
        ]:
            if candidate and candidate in rank_dirs:
                return os.path.join(hf_folder, candidate)
    except Exception:
        pass
    return None


def _find_weights_in_dir(directory: str,
                         allow_patterns: list[str]) -> tuple[list[str], bool]:
    """Find weight files in a directory and determine if safetensors."""
    files = []
    use_safetensors = False
    for pattern in allow_patterns:
        found = glob.glob(os.path.join(directory, pattern))
        if found:
            files.extend(found)
            if pattern == "*.safetensors":
                use_safetensors = True
            break
    return files, use_safetensors


def _get_load_format_config(
        load_format_str: str) -> tuple[list[str], bool, str]:
    """Get allow_patterns, use_safetensors, and index_file from load_format."""
    if load_format_str == "auto" or load_format_str == "sparse_quant":
        return ["*.safetensors", "*.bin"], False, SAFE_WEIGHTS_INDEX_NAME
    elif load_format_str in ("safetensors", "fastsafetensors"):
        return ["*.safetensors"], True, SAFE_WEIGHTS_INDEX_NAME
    elif load_format_str == "mistral":
        return (["consolidated*.safetensors"], True,
                "consolidated.safetensors.index.json")
    elif load_format_str == "pt":
        raise ValueError(
            "pt format is not supported for sparse quantized models. "
            "Please use load_format='safetensors' or 'auto'.")
    elif load_format_str == "npcache":
        return ["*.bin"], False, SAFE_WEIGHTS_INDEX_NAME
    else:
        raise ValueError(f"Unknown load_format: {load_format_str}")


@register_model_loader("sparse_quant")
class SparseQuantModelLoader(DefaultModelLoader):
    """Model loader for sparse quantized models (W8A8SC)."""

    def _prepare_weights(
        self,
        model_name_or_path: str,
        revision: Optional[str],
        fall_back_to_pt: bool,
        allow_patterns_overrides: Optional[list[str]],
    ) -> tuple[str, list[str], bool]:
        """Prepare weights for sparse quantized models."""
        model_name_or_path = (maybe_download_from_modelscope(
            model_name_or_path,
            revision,
            download_dir=self.load_config.download_dir,
            ignore_patterns=self.load_config.ignore_patterns)
                              or model_name_or_path)

        is_local = os.path.isdir(model_name_or_path)
        load_format_str = str(self.load_config.load_format).lower()
        allow_patterns, use_safetensors, index_file = _get_load_format_config(
            load_format_str)

        if allow_patterns_overrides is not None:
            allow_patterns = allow_patterns_overrides

        hf_folder = (download_weights_from_hf(
            model_name_or_path,
            self.load_config.download_dir,
            allow_patterns,
            revision,
            ignore_patterns=self.load_config.ignore_patterns)
                     if not is_local else model_name_or_path)

        # Try to find weights in rank directory first
        try:
            tp_rank = get_tensor_model_parallel_rank()
        except Exception:
            tp_rank = None

        hf_weights_files = []
        weights_in_rank_dir = False
        rank_dir = _find_rank_dir(hf_folder, tp_rank)

        if rank_dir:
            files, use_safetensors = _find_weights_in_dir(
                rank_dir, allow_patterns)
            if files:
                hf_weights_files = files
                weights_in_rank_dir = True

        # Fall back to root directory if no weights in rank dir
        if not weights_in_rank_dir:
            files, use_safetensors = _find_weights_in_dir(
                hf_folder, allow_patterns)
            hf_weights_files = files

        # Apply safetensors index filtering for non-rank-dir weights
        if use_safetensors and not weights_in_rank_dir:
            if not is_local:
                download_safetensors_index_file_from_hf(
                    model_name_or_path, index_file,
                    self.load_config.download_dir, revision)
            hf_weights_files = filter_duplicate_safetensors_files(
                hf_weights_files, hf_folder, index_file)
        elif not use_safetensors:
            hf_weights_files = filter_files_not_needed_for_inference(
                hf_weights_files)

        if not hf_weights_files:
            raise RuntimeError(
                f"Cannot find any model weights with `{model_name_or_path}`. "
                f"Patterns: {allow_patterns}, Rank dir: {rank_dir}, "
                f"TP rank: {tp_rank}, Load format: {load_format_str}")

        return hf_folder, hf_weights_files, use_safetensors
