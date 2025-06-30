#!/usr/bin/env python3
# Copyright 2025 Huawei Technologies Co., Ltd
# Copyright 2024 The vLLM team.
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
# ============================================================================

"""
refer to https://github.com/vllm-project/vllm-ascend/blob/v0.7.3/vllm_ascend/lora/punica_wrapper/punica_npu.py
"""
from typing import Callable

from mindspore import mint
from mindspore.common import dtype as mstype
from vllm_mindspore.lora.ops.torch_ops.lora_ops import (bgmv_expand, bgmv_expand_slice,
                                                        bgmv_shrink, sgmv_expand,
                                                        sgmv_expand_slice, sgmv_shrink)
from vllm.lora.punica_wrapper.punica_base import PunicaWrapperBase


# The platforms that are compatible with the PyTorch-native implementation can
# inherit this class
class PunicaWrapperNPU(PunicaWrapperBase):
    """
    PunicaWrapperNPU is designed to manage and provide metadata for the punica 
    kernel. The main function is to maintain the state information for 
    Multi-LoRA, and to provide the interface for the pytorch punica ops.
    """

    def __init__(self, max_num_batched_tokens, max_batches, device, **kwargs):
        PunicaWrapperBase.__init__(self, max_num_batched_tokens, max_batches,
                                   device)

    def _shrink_prefill(
        self,
        y,
        x,
        w_t_all,
        scale,
    ):
        sgmv_shrink(
            x,
            w_t_all,
            y,
            *self.prefill_metadata,
            scale,
        )

    def _shrink_decode(
        self,
        y,
        x,
        w_t_all,
        scale,
    ):
        bgmv_shrink(x, w_t_all, y, self.token_lora_indices, scale)

    def _expand_prefill(
        self,
        y,
        x,
        w_t_all,
        add_inputs,
    ):
        sgmv_expand(
            x,
            w_t_all,
            y,
            *self.prefill_metadata,
            add_inputs,
        )

    def _expand_decode(
        self,
        y,
        x,
        w_t_all,
        add_inputs,
    ):
        bgmv_expand(x, w_t_all, y, self.token_lora_indices, add_inputs)

    def _expand_slice_prefill(
        self,
        y,
        x,
        w_t_all,
        y_offset,
        y_slice_size,
        add_inputs,
    ):
        sgmv_expand_slice(
            x,
            w_t_all,
            y,
            *self.prefill_metadata,
            y_offset,
            y_slice_size,
            add_inputs,
        )

    def _expand_slice_decode(
        self,
        y,
        x,
        w_t_all,
        y_offset,
        y_slice_size,
        add_inputs,
    ):
        bgmv_expand_slice(x, w_t_all, y, self.token_lora_indices, y_offset,
                          y_slice_size, add_inputs)

    def _apply_expand(
        self,
        y,
        x,
        w_t_all,
        y_offset,
        y_slice_size,
        add_inputs,
    ):
        """
        Perform the ` y[:,y_offset:y_offset+y_slice_size]+=x@w_t_all` 
        computation, which is suitable for the
        GEMM of lora'b.
        """

        expand_slice_fun: Callable = (self._expand_slice_prefill
                                      if self.is_prefill else
                                      self._expand_slice_decode)
        expand_slice_fun(y, x, w_t_all, y_offset, y_slice_size, add_inputs)

    def _apply_shrink(self, y, x, w_t_all, scale):
        """
        Perform the ` y+=x@w_t_all` computation, which is suitable for the
        GEMM of lora'a.
        When `is_prefill is` true, it indicates that it is currently the
        prefill stage, and the `_shrink_prefill` function should be called.
        Otherwise, it is the decode stage, and the _shrink_decode function
        should be called.
        """
        y_org = y
        y = y.view(-1, y.shape[-1])
        shrink_fun: Callable = (self._shrink_prefill
                                if self.is_prefill else self._shrink_decode)
        shrink_fun(y, x, w_t_all, scale)
        y.view_as(y_org)

    def add_shrink(self, y, x, lora_a_stacked, scale, **kwargs):
        """
        Performs GEMM  for multiple slices of lora_a.
        When `is_prefill is` true, it indicates that it is currently the
        prefill stage, and the `_shrink_prefill` function should be called.
        Otherwise, it is the decode stage, and the _shrink_decode function
        should be called.
            
        Semantics:
        for i in range(len(lora_a_stacked)):
            y[i] += (x @ lora_a_stacked[i]) * scale
        
        Args:
            y (Union[Tuple[ms.Tensor, ...], ms.Tensor]): Output tensors
            x (ms.Tensor): Input tensor
            lora_a_stacked (Tuple[ms.Tensor, ...]): lora_a's weights
            scale (float): Scaling factor for the operation
        """

        x = x.view(-1, x.shape[-1])
        # TODO fuse these kernels
        for slice_idx in range(len(lora_a_stacked)):
            self._apply_shrink(y[slice_idx], x, lora_a_stacked[slice_idx],
                               scale)

    def add_expand(self,
                   y,
                   x,
                   lora_b_stacked,
                   lora_bias_stacked,
                   output_slices,
                   offset_start=0,
                   add_inputs=True,
                   **kwargs) -> None:
        """
        Performs GEMM and bias addition for multiple slices of lora_b.
      
        Semantics:
            for i in range(len(lora_b_stacked)):
                slice = output_slices[i]
                y[:, offset:offset+slice] += x[i] @ lora_b_stacked[i] + 
                    lora_bias_stacked[i] 
                offset += slice
            
        Args:
            y (ms.Tensor): Output tensor.
            x (Union[Tuple[ms.Tensor, ...], ms.Tensor]): Input tensors
            lora_b_stacked (Tuple[ms.Tensor, ...]): lora_b's weight
            lora_bias_stacked (Optional[Tuple[ms.Tensor, ...]]): 
                bias's weight
            output_slices (Tuple[int, ...]): Every slice's size
            add_inputs (bool):  Defaults to True.
        """
        y_org = y
        y = y.view(-1, y.shape[-1])
        offset_left = offset_start
        if lora_bias_stacked is not None:
            self._apply_bias(self.token_lora_indices, y, output_slices,
                             lora_bias_stacked)
        for slice_idx in range(len(lora_b_stacked)):
            self._apply_expand(
                y,
                x[slice_idx],
                lora_b_stacked[slice_idx],
                offset_left,
                output_slices[slice_idx],
                add_inputs=add_inputs,
            )
            offset_left += output_slices[slice_idx]
        y.view_as(y_org)

    def add_lora_embedding(self,
                           y,
                           x,
                           lora_b_stacked,
                           add_inputs=True,
                           **kwargs) -> None:
        """
        Applies lora  specifically for VocabParallelEmbeddingWithLoRA.

        Semantics:
            y += x @ lora_b_stacked

        Args:
            y (ms.Tensor): Output tensor.
            x (ms.Tensor): Input tensor.
            lora_b_stacked (ms.Tensor): lora_b's weights.
            add_inputs (bool): Default to True.
        """
        #No LoRA request, so return directly
        if self.no_lora:
            return
        # Embedding layer only need expand op
        expand_fun: Callable = (self._expand_prefill
                                if self.is_prefill else self._expand_decode)
        expand_fun(y, x, lora_b_stacked, add_inputs)

    def add_lora_linear(self,
                        y,
                        x,
                        lora_a_stacked,
                        lora_b_stacked,
                        lora_bias_stacked,
                        scale,
                        output_slices,
                        *,
                        buffer=None,
                        **kwargs) -> None:
        """
        Applicable to linear-related lora. 

        Semantics:
            for i in range(len(lora_a_stacked)):
                y[i] += (
                    x[i].unsqueeze(0)
                    @ lora_a_stacked[indices[i], layer_idx, :, :]
                    @ lora_b_stacked[indices[i], layer_idx, :, :]
                    * scale
                    ).squeeze(0)+lora_bias_stacked[i]

        Args:
            y (ms.Tensor): Output tensor. Will be changed in-place.
            x (ms.Tensor): Input tensor
            lora_a_stacked (Tuple[ms.Tensor, ...]): lora_a's weight.
            lora_b_stacked (Tuple[ms.Tensor, ...]): lora_b's weight.
            lora_bias_stacked (Optional[Tuple[ms.Tensor, ...]]): lora's bias.
            scale (float): Scaling factor.
            output_slices (Tuple[int, ...]): Every slice's size.
            buffer (Optional[Tuple[ms.Tensor, ...]]): Defaults to None.
        """
        #No LoRA request, so return directly
        if self.no_lora:
            return
        x = x.reshape(-1, x.shape[-1])
        assert len(lora_a_stacked) == len(lora_b_stacked) == len(output_slices)
        if lora_bias_stacked is not None:
            assert len(lora_bias_stacked) == len(output_slices)
            y = self._apply_bias(self.token_lora_indices, y, output_slices,
                                 lora_bias_stacked)

        if buffer is None:
            r = lora_b_stacked[0].shape[-1]
            # We set the buffer to be float32 by default, consistent with the
            # triton op
            buffer = tuple(
                mint.zeros((x.shape[0], r), dtype=mstype.float32)
                for _ in range(len(output_slices)))
        self.add_shrink(buffer, x, lora_a_stacked, scale, **kwargs)
        self.add_expand(y,
                        buffer,
                        lora_b_stacked,
                        None,
                        output_slices,
                        add_inputs=True,
                        **kwargs)

    def add_lora_logits(self,
                        y,
                        x,
                        lora_a_stacked,
                        lora_b_stacked,
                        scale,
                        *,
                        buffer=None,
                        **kwargs) -> None:
        """
        Applies lora  specifically for LogitsProcessorWithLoRA.
        
        Semantics:
            buffer = (x @ lora_a_stacked) * scale
            y += buffer @ lora_b_stacked

        Args:
            y (ms.Tensor): Output tensor.
            x (ms.Tensor): Input tensor.
            lora_a_stacked (ms.Tensor): lora_a's weights.
            lora_b_stacked (ms.Tensor):lora_b's weights.
            scale (float): Scaling factor.
            buffer (Optional[ms.Tensor]):Default to None.
        """
        #No LoRA request, so return directly
        if self.no_lora:
            return
        y_org = y
        y = y.view(-1, y.shape[-1])
        x = x.view(-1, x.shape[-1])
        r = lora_b_stacked.shape[-1]
        if buffer is None:
            # We set the buffer to be float32 by default, consistent with the
            # triton op
            buffer = mint.zeros((x.shape[0], r), dtype=mstype.float32)
        # LogitsProcessorWithLoRA always using bgmv.
        bgmv_shrink(x, lora_a_stacked, buffer, self.sampler_indices, scale)
        bgmv_expand(buffer,
                    lora_b_stacked,
                    y,
                    self.sampler_indices,
                    add_inputs=True)
        y.view_as(y_org)
