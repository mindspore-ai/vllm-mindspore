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

import os
from abc import abstractmethod
from collections.abc import Iterable
from typing import Any, Optional, Union, cast

import mindspore as ms
import numpy as np
from mindspore import Tensor, mutable, nn
from mindspore.common import dtype as mstype
from vllm.attention.backends.abstract import AttentionType
from vllm.attention.selector import get_attn_backend
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import get_dp_group, get_ep_group
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.sequence import IntermediateTensors
from vllm.v1.sample.metadata import SamplingMetadata

from vllm_mindspore.model_executor.models.attention_mask import (
    LowerTriangularMask)
from vllm_mindspore.model_executor.models.interfaces import (
    is_mixture_of_experts, supports_moe_dp_tp)
from vllm_mindspore.model_executor.models.utils import (convert_pin,
                                                        is_use_ringmla)
from vllm_mindspore.model_executor.utils import set_model_context
from vllm_mindspore.utils import (STR_DTYPE_TO_MS_DTYPE, create_kv_cache,
                                  is_310p)
from vllm_mindspore.v1.attention.backends.ms_attn import MsAttentionMetadata


class AttentionWrapper(AttentionLayerBase):

    def __init__(self):
        super().__init__()
        vllm_config = get_current_vllm_config()
        block_size = vllm_config.cache_config.block_size
        num_kv_heads = vllm_config.model_config.get_num_kv_heads(
            vllm_config.parallel_config)
        head_size = vllm_config.model_config.get_head_size()
        num_block = 0
        self.kv_shape = [num_block, block_size, num_kv_heads, head_size]
        self.kv_cache = [
            (create_kv_cache(self.kv_shape, vllm_config.model_config.dtype),
             create_kv_cache(self.kv_shape, vllm_config.model_config.dtype))
            for _ in range(vllm_config.parallel_config.pipeline_parallel_size)
        ]

        self.attn_type = AttentionType.DECODER

        # add for v1
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.dtype = vllm_config.model_config.dtype
        self.block_size = block_size
        self.sliding_window = None
        self.kv_sharing_target_layer_name = None
        self.attn_backend = get_attn_backend(head_size,
                                             vllm_config.model_config.dtype,
                                             vllm_config.model_config.dtype,
                                             block_size, False,
                                             vllm_config.model_config.use_mla,
                                             False)

    def get_attn_backend(self):
        return self.attn_backend


class MLAAttentionWrapper(AttentionWrapper):

    def __init__(self, fa3_quant=False, kv_cache_dtype=None):
        super().__init__()
        vllm_config = get_current_vllm_config()
        self.use_ringmla = is_use_ringmla(vllm_config)
        if kv_cache_dtype is None:
            kv_cache_dtype = vllm_config.model_config.dtype
        self.dtype = kv_cache_dtype
        if not self.use_ringmla:
            self.kv_cache = [
                (
                    ms.mint.zeros(
                        self.kv_shape,  # type: ignore[misc]
                        dtype=vllm_config.model_config.dtype), ) for _ in
                range(vllm_config.parallel_config.pipeline_parallel_size)
            ]
        else:
            kv_lora_rank = getattr(vllm_config.model_config.hf_text_config,
                                   'kv_lora_rank', 0)
            qk_rope_head_dim = getattr(vllm_config.model_config.hf_text_config,
                                       'qk_rope_head_dim', 0)
            # k_shape, r_shape used for mla_op
            if fa3_quant:
                k_shape = [*(self.kv_shape[0:-2]), kv_lora_rank]
                r_shape = [*(self.kv_shape[0:-2]), qk_rope_head_dim]
                self.kv_cache = [(
                    ms.mint.zeros(k_shape, dtype=kv_cache_dtype),
                    ms.mint.zeros(r_shape,
                                  dtype=vllm_config.model_config.dtype),
                ) for _ in range(
                    vllm_config.parallel_config.pipeline_parallel_size)]
            else:
                k_shape = [*(self.kv_shape[0:-1]), kv_lora_rank]
                r_shape = [*(self.kv_shape[0:-1]), qk_rope_head_dim]
                self.kv_cache = [
                    (ms.mint.zeros(k_shape,
                                   dtype=vllm_config.model_config.dtype),
                     ms.mint.zeros(r_shape,
                                   dtype=vllm_config.model_config.dtype))
                    for _ in range(
                        vllm_config.parallel_config.pipeline_parallel_size)
                ]


class MsModelBase:

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        lora_config = vllm_config.lora_config

        self.config = config
        self.model_config = vllm_config.model_config
        self.lora_config = lora_config
        self.cache_config = vllm_config.cache_config
        self.parallel_config = vllm_config.parallel_config
        self.load_config = vllm_config.load_config
        self.scheduler_config = vllm_config.scheduler_config

        # aclgraph need construct block table with real input
        # here must know max_model_len and block_size
        # for speculative infer, this value need update
        self.block_size = self.cache_config.block_size
        self.max_model_len = vllm_config.model_config.max_model_len
        self.max_block_num = int(self.max_model_len / self.block_size)

        self.modules_dict: Optional[dict[str, nn.Cell]] = None

        self.enable_chunked_prefill = (
            vllm_config.scheduler_config.enable_chunked_prefill)
        self.enable_prefix_caching = (
            vllm_config.cache_config.enable_prefix_caching)
        self.num_layers = self.model_config.get_num_layers(
            self.parallel_config)

        self.use_ringmla: bool = False
        self.has_prefill_warmup: bool = False
        self.has_chunked_warmup: bool = not self.use_ringmla
        self.kv_caches: list[Any] = []
        self.casual_mask = LowerTriangularMask(
            dtype=self.model_config.dtype,
            max_model_len=self.model_config.max_model_len)
        self.model: Optional[nn.Cell] = None
        self.lm_head: Optional[nn.Cell] = None

    def get_model_path(self):
        model_name_or_path = self.model_config.model
        if os.path.isdir(model_name_or_path):
            return model_name_or_path
        else:
            from vllm.model_executor.model_loader.weight_utils import (
                download_weights_from_hf)
            allow_patterns = ["*.safetensors"]
            revision = self.model_config.revision
            return download_weights_from_hf(
                model_name_or_path,
                self.load_config.download_dir,
                allow_patterns,
                revision,
                ignore_patterns=self.load_config.ignore_patterns,
            )

    def set_modules(self, model_dicts: dict[str, nn.Cell]):
        self.modules_dict = model_dicts

    def _check_modules_valid(self):
        if self.modules_dict is None:
            raise RuntimeError("Should set modules firstly!")

    def named_parameters(self):
        self._check_modules_valid()
        modules_dict = cast(dict[str, nn.Cell], self.modules_dict)
        for cell_name, module in modules_dict.items():
            for par_name, par in module.parameters_and_names():
                if cell_name != "self":
                    par_name = cell_name + "." + par_name

                yield par_name, par

    def get_params_dict(self):
        self._check_modules_valid()
        modules_dict = cast(dict[str, nn.Cell], self.modules_dict)
        params_dict = dict()
        for name, module in modules_dict.items():
            module_params = module.parameters_dict()
            if name != "self":
                new_module_params = dict()
                for param_name, param in module_params.items():
                    new_module_params[name + "." + param_name] = param
                module_params = new_module_params
            params_dict.update(module_params)

        return params_dict

    def named_modules(self, remove_duplicate: bool = True):
        self._check_modules_valid()
        modules_dict = cast(dict[str, nn.Cell], self.modules_dict)
        for name, module in modules_dict.items():
            for module_name, sub_module in module.cells_and_names():
                if name != "self":
                    module_name = name + "." + module_name
                yield module_name, sub_module

    def get_submodule(self, target: str):
        parts = target.split(".")
        if target == "":
            return self
        for part in parts:
            if not part:
                raise ValueError(
                    f"Invalid submodule path: empty part in '{target}'")
        current = self
        for part in parts:
            current = getattr(current, part)
        return current

    def eval(self):
        self._check_modules_valid()
        modules_dict = cast(dict[str, nn.Cell], self.modules_dict)
        for _, module in modules_dict.items():
            module.set_train(False)

        return self

    def __call__(
        self,
        input_ids: Tensor,
        positions: Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[Tensor] = None,
        previous_hidden_states: Optional[Tensor] = None,
        spec_step_idx: int = 0,
        **kwargs,
    ) -> Union[Tensor, IntermediateTensors]:
        return self.forward(input_ids,
                            positions,
                            intermediate_tensors,
                            inputs_embeds,
                            previous_hidden_states=previous_hidden_states,
                            spec_step_idx=spec_step_idx,
                            **kwargs)

    def forward(self,
                input_ids: Tensor,
                positions: Tensor,
                intermediate_tensors: Optional[IntermediateTensors] = None,
                inputs_embeds: Optional[Tensor] = None,
                **kwargs) -> Union[Tensor, IntermediateTensors]:
        raise NotImplementedError

    def get_kvcache(self):
        key_cache = []
        value_cache = []
        forward_context = get_forward_context()
        for i in range(self.num_layers):
            k_cache = self.kv_caches[i].kv_cache[
                forward_context.virtual_engine][0]
            v_cache = self.kv_caches[i].kv_cache[
                forward_context.virtual_engine][1]
            key_cache.append(k_cache)
            value_cache.append(v_cache)
        return mutable(key_cache), mutable(value_cache)

    @abstractmethod
    def compute_logits(
        self,
        hidden_states: Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[Tensor]:
        raise NotImplementedError(
            "Function compute_logits should be Implemented!")

    @abstractmethod
    def load_weights(self, weights: Iterable[tuple[str, Tensor]]) -> set[str]:
        raise NotImplementedError(
            "Function load_weights should be Implemented!")

    def _dummy_attention_metadata(self, input_ids: Tensor, positions: Tensor):
        if input_ids is not None:
            input_len = input_ids.shape[0]
        elif positions is not None:
            # input_ids is None in multi modal model with v1 arch
            input_len = positions.shape[-1]

        max_seq_len = ms.Tensor(input_len, dtype=ms.int32)
        seq_lengths = ms.Tensor([input_len], dtype=ms.int32)
        q_seq_lens_np = np.array([input_len], dtype=np.int32)
        seq_lens_np = np.array([input_len], dtype=np.int32)
        # context len is 0 for prefill, and 1 for chunked and decode.
        context_lens_tensor = ms.Tensor([0], dtype=ms.int32) if not (
            self.has_prefill_warmup) else ms.Tensor([1], dtype=ms.int32)
        block_tables = ms.Tensor([[0] * self.max_block_num], dtype=ms.int32)
        slot_mapping = [-1 for _ in range(input_len)]
        slot_mapping = ms.Tensor(slot_mapping, dtype=ms.int32)
        return MsAttentionMetadata(
            num_actual_tokens=input_len,
            max_query_len=input_len,
            max_seq_len=max_seq_len,
            seq_lens=seq_lengths,
            seq_lens_np=seq_lens_np,
            block_tables=block_tables,
            slot_mapping=slot_mapping,
            q_seq_lens_np=q_seq_lens_np,
            context_lens=context_lens_tensor,
            # To enforce prefill and decode are both complied in warmup process.
            # So set max_context_lens to 0 for prefill and 1 for decode.
            max_context_lens=0 if not self.has_prefill_warmup else 1,
        )

    def prepare_base_inputs(self, input_ids, positions):
        attn_metadata = get_forward_context().attn_metadata
        # 0.9.1 attn_metadata[layer_name], don't have layer_name here
        # so we just take one by default
        if isinstance(attn_metadata, dict) and '1' in attn_metadata:
            attn_metadata = attn_metadata['1']
        if attn_metadata is None:
            attn_metadata = self._dummy_attention_metadata(
                input_ids, positions)
        key_cache, value_cache = self.get_kvcache()

        is_prefill = attn_metadata.max_context_lens == 0
        query_lens_np = attn_metadata.q_seq_lens_np
        seq_lens_np = attn_metadata.seq_lens_np

        if input_ids is not None:
            input_ids = input_ids.astype(ms.int32)
        q_seq_lens = ms.Tensor(query_lens_np, dtype=ms.int32)
        position_ids = ms.Tensor(positions, dtype=ms.int32)
        attention_mask = self.casual_mask.gen_attention_mask(
            is_prefill, position_ids, query_lens_np, seq_lens_np)

        model_inputs = {}
        model_inputs["input_ids"] = convert_pin(input_ids)
        model_inputs["batch_valid_length"] = convert_pin(
            ms.from_numpy(seq_lens_np))
        model_inputs["block_tables"] = convert_pin(attn_metadata.block_tables)
        model_inputs["slot_mapping"] = convert_pin(attn_metadata.slot_mapping)
        model_inputs["position_ids"] = convert_pin(position_ids)
        model_inputs["q_seq_lens"] = convert_pin(q_seq_lens)
        model_inputs["attention_mask"] = convert_pin(attention_mask)
        model_inputs["key_cache"] = key_cache
        model_inputs["value_cache"] = value_cache

        return model_inputs, is_prefill


class NativeModel(MsModelBase):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.quant_config = vllm_config.quant_config
        if vllm_config.lora_config is not None:
            # native model lora only support pynative mode now
            vllm_config.model_config.enforce_eager = True
        self.is_eager_mode = vllm_config.model_config.enforce_eager
        self.prefill_graph = None
        self.decode_graph = None

        if is_mixture_of_experts(self):
            ep_size = get_ep_group().world_size
            custom_ep_size = \
                vllm_config.additional_config.get("expert_parallel", ep_size) \
                if vllm_config.additional_config is not None else ep_size
            pure_ep = self.parallel_config.enable_expert_parallel and \
                (ep_size // int(custom_ep_size)) == 1
            if get_dp_group().world_size > 1 and not pure_ep:
                if not supports_moe_dp_tp(self):
                    raise ValueError(
                        "The MoE Model do not support Data Parallel mix with "
                        "Tensor Parallel. If want to use MoE with DP + TP, "
                        "please implement `supports_moe_dp_tp` in models.")
                self.moe_dp_need_pad = True
                self.dp_group = get_dp_group().device_group._name
                self.dp_cpu_group = get_dp_group().cpu_group._name
                self.dp_world_size = get_dp_group().world_size
                self.dp_rank = get_dp_group().rank_in_group
            else:
                self.moe_dp_need_pad = False

    @property
    def ready_model(self) -> nn.Cell:
        if self.model is None:
            raise RuntimeError("Model not initialized")
        return self.model

    @property
    def ready_lm_head(self) -> nn.Cell:
        if self.lm_head is None:
            raise RuntimeError("lm head not initialized")
        return self.lm_head

    def common_preprocess(self, vllm_config, prefix=""):
        self.set_modules({"model": self.model, "lm_head": self.lm_head})

        self.casual_mask = LowerTriangularMask(
            dtype=self.model_config.dtype,
            max_model_len=self.model_config.max_model_len)
        self.kv_caches = [AttentionWrapper() for i in range(self.num_layers)]

        compilation_config = vllm_config.compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        for i in range(self.num_layers):
            compilation_config.static_forward_context[str(
                i)] = self.kv_caches[i]

    def set_model_inputs(self,
                         input_ids=None,
                         position_ids=None,
                         intermediate_tensors=None,
                         inputs_embeds=None):
        if input_ids is None:
            dyn_input_ids = None
        else:
            dyn_input_ids = ms.Tensor(shape=[None] * input_ids.ndim,
                                      dtype=mstype.int32)

        if position_ids is None:
            dyn_position_ids = None
        else:
            dyn_position_ids = ms.Tensor(shape=[None] * position_ids.ndim,
                                         dtype=mstype.int32)

        if inputs_embeds is None:
            dyn_inputs_embeds = None
        else:
            dyn_inputs_embeds = ms.Tensor(shape=[None] * inputs_embeds.ndim,
                                          dtype=inputs_embeds.dtype)

        if intermediate_tensors is None:
            dyn_intermediate_hidden_states = None
            dyn_intermediate_residual = None
        else:
            dyn_input_ids = None
            dyn_intermediate_hidden_states = ms.Tensor(
                shape=[None] * intermediate_tensors["hidden_states"].ndim,
                dtype=intermediate_tensors["hidden_states"].dtype)
            dyn_intermediate_residual = ms.Tensor(
                shape=[None] * intermediate_tensors["residual"].ndim,
                dtype=intermediate_tensors["residual"].dtype)

        block_size = self.cache_config.block_size
        num_kv_heads = self.model_config.get_num_kv_heads(self.parallel_config)
        head_size = self.model_config.get_head_size()
        # FormatCast op is used to convert kv_cache to nz format for 310p.
        # The kv shape is flattened to 3 dimensions,
        # because FormatCast op only support 3d input.
        kv_cache_shape = (None, block_size, num_kv_heads * head_size) \
            if is_310p() else (None, block_size, num_kv_heads, head_size)

        kv_cache_dtype = (self.model_config.dtype
                          if self.cache_config.cache_dtype == "auto" else
                          self.cache_config.cache_dtype)
        if kv_cache_dtype in STR_DTYPE_TO_MS_DTYPE:
            kv_cache_dtype = STR_DTYPE_TO_MS_DTYPE[kv_cache_dtype]

        num_layers = self.model_config.get_num_layers(self.parallel_config)

        dyn_key_cache = Tensor(shape=kv_cache_shape, dtype=kv_cache_dtype)
        dyn_value_cache = Tensor(shape=kv_cache_shape, dtype=kv_cache_dtype)
        dyn_key_caches = mutable([dyn_key_cache for _ in range(num_layers)])
        dyn_value_caches = mutable(
            [dyn_value_cache for _ in range(num_layers)])

        dyn_slot_mapping = Tensor(shape=[None], dtype=mstype.int32)
        dynamic_attention_mask = Tensor(shape=[None, None],
                                        dtype=self.model_config.dtype)
        dyn_batch_valid_length = Tensor(shape=[None], dtype=mstype.int32)
        dyn_q_seq_lens = Tensor(shape=[None], dtype=mstype.int32)
        dyn_block_tables = Tensor(shape=[None, None], dtype=mstype.int32)

        # Build set_inputs arguments
        set_inputs_args = [
            dyn_input_ids, dyn_position_ids, dyn_key_caches, dyn_value_caches,
            dyn_slot_mapping, dynamic_attention_mask, dyn_batch_valid_length,
            dyn_q_seq_lens, dyn_block_tables, dyn_intermediate_hidden_states,
            dyn_intermediate_residual, dyn_inputs_embeds
        ]
        # Add MoE-specific parameters if available
        if moe_params := self._get_moe_input_params():
            set_inputs_args.extend([
                moe_params['dp_pad_index'], moe_params['dp_unpad_index'],
                moe_params['dp_pad_index_with_offset'],
                moe_params['dp_unpad_index_total_with_offset']
            ])
        # Add extra parameters if available
        # (e.g., deepstack_input_embeds for multimodal models)
        if extra_params := self._get_extra_input_params():
            set_inputs_args.extend(extra_params)

        self.ready_model.set_inputs(*set_inputs_args)

        dynamic_hidden_states = Tensor(shape=[None, None],
                                       dtype=self.model_config.dtype)
        self.ready_lm_head.set_inputs(dynamic_hidden_states)

    def _get_moe_input_params(self):
        """Get MoE-specific input parameters.
        
        Returns a dict with MoE parameters if MoE is supported, None otherwise.
        Subclasses can override this method to customize MoE parameter handling.
        
        Returns:
            dict or None: Dict with keys: dp_pad_index, dp_unpad_index,
                dp_pad_index_with_offset, dp_unpad_index_total_with_offset
                if MoE is supported, None otherwise.
        """
        if supports_moe_dp_tp(self):
            return {
                'dp_pad_index': (Tensor(shape=[None], dtype=mstype.int32)
                                 if self.moe_dp_need_pad else None),
                'dp_unpad_index': (Tensor(shape=[None], dtype=mstype.int32)
                                   if self.moe_dp_need_pad else None),
                'dp_pad_index_with_offset':
                (Tensor(shape=[None], dtype=mstype.int32)
                 if self.moe_dp_need_pad else None),
                'dp_unpad_index_total_with_offset':
                (Tensor(shape=[None], dtype=mstype.int32)
                 if self.moe_dp_need_pad else None),
            }
        return None

    def _get_extra_input_params(self):
        """Get extra input parameters for model-specific features.
        
        Subclasses can override this method to add additional parameters
        (e.g., deepstack_input_embeds for multimodal models).
        
        Returns:
            list or None: List of additional parameters to pass to set_inputs,
                         or None if no extra parameters are needed.
        """
        return None

    def prepare_moe_dp_tp_inputs(self):
        """
        prepare moe padding indices to support moe dp + tp.
        """
        if self.moe_dp_need_pad:
            dp_meta = get_forward_context().dp_metadata
            token_num_total = dp_meta.num_tokens_across_dp_cpu.asnumpy()
            max_token_num = dp_meta.max_tokens_across_dp_cpu
            token_num_total_cumsum = token_num_total.cumsum()
            max_token_num = max_token_num.numpy()

            token_num_total = np.diff(token_num_total_cumsum, prepend=0)
            total_pad_num = max_token_num - token_num_total
            this_pad_num = total_pad_num[self.dp_rank]

            dp_unpad_index = np.arange(token_num_total[self.dp_rank])
            dp_pad_index = np.pad(dp_unpad_index, (0, this_pad_num))

            dp_pad_index_total_with_offset = [
                np.pad(
                    np.arange(
                        0 if rank == 0 else token_num_total_cumsum[rank - 1],
                        token_num_total_cumsum[rank]),
                    (0, total_pad_num[rank]))
                for rank in range(self.dp_world_size)
            ]
            dp_pad_index_total_with_offset = \
                np.concatenate(dp_pad_index_total_with_offset, axis=0)

            dp_unpad_index_total_with_offset = [
                np.arange(token_num_total[rank]) + rank * max_token_num
                for rank in range(self.dp_world_size)
            ]
            dp_unpad_index_total_with_offset = \
                np.concatenate(dp_unpad_index_total_with_offset, axis=0)

            dp_unpad_index = ms.from_numpy(dp_unpad_index.astype(np.int32))
            dp_pad_index = ms.from_numpy(dp_pad_index.astype(np.int32))
            dp_pad_index_total_with_offset = ms.from_numpy(
                dp_pad_index_total_with_offset.astype(np.int32))
            dp_unpad_index_total_with_offset = ms.from_numpy(
                dp_unpad_index_total_with_offset.astype(np.int32))
        else:
            dp_unpad_index = None
            dp_pad_index = None
            dp_pad_index_total_with_offset = None
            dp_unpad_index_total_with_offset = None

        return (dp_unpad_index, dp_pad_index, dp_pad_index_total_with_offset,
                dp_unpad_index_total_with_offset)

    def prepare_inputs(self, input_ids, positions, intermediate_tensors,
                       inputs_embeds):
        model_inputs, is_prefill = self.prepare_base_inputs(
            input_ids, positions)

        new_model_inputs = {}
        new_model_inputs["input_ids"] = model_inputs["input_ids"]
        new_model_inputs["batch_valid_length"] = model_inputs[
            "batch_valid_length"]
        new_model_inputs["block_tables"] = model_inputs["block_tables"]
        new_model_inputs["slot_mapping"] = model_inputs["slot_mapping"]
        new_model_inputs["positions"] = model_inputs["position_ids"]
        new_model_inputs["q_seq_lens"] = model_inputs["q_seq_lens"]
        new_model_inputs["attn_mask"] = model_inputs["attention_mask"]
        new_model_inputs["key_caches"] = model_inputs["key_cache"]
        new_model_inputs["value_caches"] = model_inputs["value_cache"]
        # for multimodal model
        if intermediate_tensors is not None:
            new_model_inputs["input_ids"] = None
            new_model_inputs[
                "intermediate_hidden_states"] = intermediate_tensors[
                    "hidden_states"]
            new_model_inputs["intermediate_residual"] = intermediate_tensors[
                "residual"]
        else:
            new_model_inputs["intermediate_hidden_states"] = None
            new_model_inputs["intermediate_residual"] = None
        new_model_inputs["inputs_embeds"] = inputs_embeds

        if supports_moe_dp_tp(self):
            dp_unpad_index, dp_pad_index, dp_pad_index_total_with_offset, \
            dp_unpad_index_total_with_offset = self.prepare_moe_dp_tp_inputs()
            new_model_inputs["dp_unpad_index"] = dp_unpad_index
            new_model_inputs["dp_pad_index"] = dp_pad_index
            new_model_inputs["dp_pad_index_total_with_offset"] = \
                dp_pad_index_total_with_offset
            new_model_inputs["dp_unpad_index_total_with_offset"] = \
                dp_unpad_index_total_with_offset

        return new_model_inputs, is_prefill

    def exec_model(self,
                   input_ids: Tensor,
                   positions: Tensor,
                   intermediate_tensors: IntermediateTensors = None,
                   inputs_embeds: Tensor = None,
                   **kwargs):
        model_inputs, is_prefill = self.prepare_inputs(input_ids, positions,
                                                       intermediate_tensors,
                                                       inputs_embeds)
        model_inputs.update(kwargs)
        # for dummy_attention_metadata
        if is_prefill and not self.has_prefill_warmup:
            self.has_prefill_warmup = True

        # eager mode
        if self.is_eager_mode:
            set_model_context("is_prefill", is_prefill)
            model_output = self.model(**model_inputs)
            return model_output

        # graph mode
        if is_prefill:
            self.model.phase = "prefill"
            if self.prefill_graph is None:
                set_model_context("is_prefill", True)
                self.model._set_jit_graph_name("prefill")
                self.set_model_inputs(input_ids, positions,
                                      intermediate_tensors, inputs_embeds)
                self.prefill_graph = ms.jit(function=self.model,
                                            jit_level="O0")
            model_output = self.prefill_graph(**model_inputs)
        else:
            self.model.phase = "increment"
            if self.decode_graph is None:
                set_model_context("is_prefill", False)
                self.model._set_jit_graph_name("decode")
                self.set_model_inputs(input_ids, positions,
                                      intermediate_tensors, inputs_embeds)
                self.decode_graph = ms.jit(function=self.model, jit_level="O0")
            model_output = self.decode_graph(**model_inputs)

        return model_output
