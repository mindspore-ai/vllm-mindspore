#!/usr/bin/env python3
# Copyright 2025 Huawei Technologies Co., Ltd
# Copyright 2024 The vLLM team.
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

import os
from typing import Dict, List, Optional, Union

import safetensors.torch
import torch
from vllm.lora.lora import LoRALayerWeights
from vllm.lora.peft_helper import PEFTHelper
from vllm.lora.utils import is_regex_target_modules, parse_fine_tuned_lora_name
from vllm.model_executor.models.utils import WeightsMapper
from vllm.utils import is_pin_memory_available

from vllm_mindspore.lora.layers import BaseLayerWithLoRA

_GLOBAL_LORA_ID = 0


def get_lora_id():
    global _GLOBAL_LORA_ID
    _GLOBAL_LORA_ID += 1
    return _GLOBAL_LORA_ID


def register_module(self, module_name: str, module: "BaseLayerWithLoRA"):
    assert isinstance(module, BaseLayerWithLoRA)
    self.modules[module_name] = module


@classmethod  #type:ignore
def from_lora_tensors(
    cls,
    lora_model_id: int,
    tensors: Dict[str, torch.Tensor],
    peft_helper: PEFTHelper,
    device: str = "cuda",
    dtype: Optional[torch.dtype] = None,
    embeddings: Optional[Dict[str, torch.Tensor]] = None,
    target_embedding_padding: Optional[int] = None,
    embedding_modules: Optional[Dict[str, str]] = None,
    embedding_padding_modules: Optional[List[str]] = None,
    weights_mapper: Optional[WeightsMapper] = None,
):
    """Create a LoRAModel from a dictionary of tensors."""
    pin_memory = str(device) == "cpu" and is_pin_memory_available()
    loras: Dict[str, LoRALayerWeights] = {}
    for tensor_name, tensor in tensors.items():
        module_name, is_lora_a, is_bias = parse_fine_tuned_lora_name(
            tensor_name, weights_mapper)
        if module_name not in loras:
            lora_embeddings_tensor = None
            if embeddings:
                assert embedding_modules is not None
                embeddings_module = next(
                    (k for k in embedding_modules if k in module_name), None)
                if embeddings_module:
                    lora_embeddings_tensor = embeddings[
                        embedding_modules[embeddings_module]]
                    if pin_memory:
                        lora_embeddings_tensor = (
                            lora_embeddings_tensor.pin_memory())
            loras[module_name] = LoRALayerWeights.from_config(
                module_name, peft_helper, lora_embeddings_tensor)

        if is_bias:
            # vllm-mindspore remove tensor device
            loras[module_name].bias = tensor.to(dtype=dtype).t()
            bias = tensor.to(dtype=dtype).t()
            if pin_memory:
                bias = bias.pin_memory()
            loras[module_name].bias = bias
        elif is_lora_a:
            loras[module_name].lora_a = tensor.to(dtype=dtype).t()
            if pin_memory:
                loras[module_name].lora_a = loras[
                    module_name].lora_a.pin_memory()
        else:
            loras[module_name].lora_b = tensor.to(dtype=dtype).t()
            assert embedding_padding_modules is not None
            if any(name in module_name for name in embedding_padding_modules
                   ) and target_embedding_padding is not None:
                lora_b = loras[module_name].lora_b
                assert target_embedding_padding >= lora_b.shape[1]
                addition = target_embedding_padding - lora_b.shape[1]
                loras[module_name].lora_b = torch.nn.functional.pad(
                    lora_b, (0, addition))
            if pin_memory:
                loras[module_name].lora_b = loras[
                    module_name].lora_b.pin_memory()

    for lora in loras.values():
        lora.optimize()

    return cls(lora_model_id,
               peft_helper.r,
               loras,
               scaling_factor=peft_helper.vllm_long_context_scaling_factor)


@classmethod  #type:ignore
def from_local_checkpoint(
    cls,
    lora_dir: str,
    expected_lora_modules: List[str],
    peft_helper: PEFTHelper,
    *,
    lora_model_id: Optional[int] = None,
    device: str = "cuda",
    dtype: Optional[torch.dtype] = None,
    target_embedding_padding: Optional[int] = None,
    embedding_modules: Optional[Dict[str, str]] = None,
    embedding_padding_modules: Optional[List[str]] = None,
    weights_mapper: Optional[WeightsMapper] = None,
):
    """Create a LoRAModel from a local checkpoint.
        
        Args:
            lora_dir: The local path that has lora data.
            expected_lora_modules: Name of modules that are expected to be
                replaced by lora.
            peft_helper: Loaded lora configuration information.
            lora_model_id: Lora model id. If not given, automatically set by
                a global counter.
            device: Device where the lora model is loaded.
            dtype: dtype of the lora model weights.

        Returns:
            Loaded LoRA Model.
        """
    lora_tensor_path = os.path.join(lora_dir, "adapter_model.safetensors")
    lora_bin_file_path = os.path.join(lora_dir, "adapter_model.bin")
    new_embeddings_tensor_path = os.path.join(lora_dir,
                                              "new_embeddings.safetensors")
    new_embeddings_bin_file_path = os.path.join(lora_dir, "new_embeddings.bin")

    unexpected_modules: List[Union[list[str], str]]
    if os.path.isfile(lora_tensor_path):
        tensors: Dict[str, torch.Tensor] = {}
        # Find unexpected modules.
        # Use safetensor key as a source of truth to find expected modules.
        # in peft if you have target_modules A, B, C and C does not exist
        # in the model it won’t error and model will be trained with A, B
        # loraified. C won’t exist in the safetensor but it will exist in
        # the target_modules of the adapter_config.json.
        unexpected_modules = []
        # vllm-mindspore safetensors open with np
        with safetensors.safe_open(lora_tensor_path,
                                   framework="np") as f:  # type: ignore
            for lora_module in f.keys():  # noqa
                module_name, _, _ = parse_fine_tuned_lora_name(
                    lora_module, weights_mapper)
                part_name = module_name.split(".")[-1]
                if part_name not in expected_lora_modules:
                    unexpected_modules.append(module_name)
            if unexpected_modules:
                raise ValueError(
                    f"While loading {lora_dir}, expected"
                    f" target modules in {expected_lora_modules}"
                    f" but received {unexpected_modules}."
                    f" Please verify that the loaded LoRA module is correct")
            # Load tensors if there are only expected modules.
            for module in f.keys():  # noqa
                # vllm-mindspore add numpy to tensor
                tensors[module] = torch.Tensor(f.get_tensor(module))
    elif os.path.isfile(lora_bin_file_path):
        # When a bin file is provided, we rely on config to find unexpected
        # modules.
        unexpected_modules = []
        target_modules = peft_helper.target_modules
        if not isinstance(target_modules, list):
            target_modules = [target_modules]
        for module in target_modules:
            # Compatible with more modules,
            # such as:layers.11.self_attn.k_proj
            part_name = module.split(".")[-1]
            if part_name not in expected_lora_modules:
                unexpected_modules.append(module)
        # loaded lora's target modules must be a subset of
        # expected_lora_modules. It is not reliable. See
        # https://github.com/vllm-project/vllm/pull/5909. But there's no
        # other better mechanism.
        if unexpected_modules and not is_regex_target_modules(
                peft_helper.target_modules, expected_lora_modules):
            raise ValueError(
                f"While loading {lora_dir}, expected"
                f" target modules in {expected_lora_modules}"
                f" but received {unexpected_modules}."
                f" Please verify that the loaded LoRA module is correct")
        tensors = torch.load(lora_bin_file_path, map_location=device)
    else:
        raise ValueError(f"{lora_dir} doesn't contain tensors")

    embeddings = None
    if os.path.isfile(new_embeddings_tensor_path):
        embeddings = safetensors.torch.load_file(new_embeddings_tensor_path)
    elif os.path.isfile(new_embeddings_bin_file_path):
        embeddings = torch.load(new_embeddings_bin_file_path,
                                map_location=device,
                                weights_only=True)

    return cls.from_lora_tensors(
        lora_model_id=get_lora_id()
        if lora_model_id is None else lora_model_id,
        tensors=tensors,
        peft_helper=peft_helper,
        device=device,
        dtype=dtype,
        embeddings=embeddings,
        target_embedding_padding=target_embedding_padding,
        embedding_modules=embedding_modules,
        embedding_padding_modules=embedding_padding_modules,
        weights_mapper=weights_mapper)
