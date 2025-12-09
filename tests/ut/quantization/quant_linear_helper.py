# SPDX-License-Identifier: Apache-2.0
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
"""a helper designed for test."""

from mindspore import jit, nn

from vllm_mindspore.model_executor.layers.linear import ReplicatedLinear
from vllm_mindspore.model_executor.model_loader.weight_utils import (
    default_weight_loader)


class QuantLinearHelper(nn.Cell):
    """A helper designed for test."""

    def __init__(self, quant_config, hf_config, dtype):
        super().__init__()
        self.linear = self._build_linears(quant_config, hf_config, dtype)
        self.linear.quant_method.process_weights_after_loading(self.linear)

    def _build_linears(self, quant_config, hf_config, dtype):
        """Build a ReplicatedLinear instance."""
        return ReplicatedLinear(hf_config.hidden_size,
                                hf_config.hidden_size,
                                bias=hf_config.attention_bias,
                                params_dtype=dtype,
                                quant_config=quant_config)

    def load_weights_into_linear(self, weights):
        """Load weights into linear layer."""
        params = self.parameters_dict()
        loaded = []
        for para_name, para in weights.items():
            param = params.get(para_name)
            if param is None:
                continue
            loaded.append(para_name)
            weight_loader = getattr(param, "weight_loader",
                                    default_weight_loader)
            weight_loader(param, para)

        print(f"weights not use: {set(weights.keys()) - set(loaded)}",
              flush=True)
        print(f"params not load: {set(params.keys()) - set(loaded)}",
              flush=True)

    @jit(jit_config={"infer_boost": "on", "jit_level": "O0"})
    def construct(self, x):
        return self.linear(x)
