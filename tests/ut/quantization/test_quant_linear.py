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
"""Run quant linear accuracy test"""
import vllm_mindspore

import glob
import json
import os
import tempfile

import mindspore
import numpy as np
import pytest
from unittest.mock import patch
from mindspore import Tensor, mint
from mindspore.common import dtype as mstype
from safetensors import safe_open

from tests.ut.quantization.numpy_quantizer import NumpyQuantizer
from tests.ut.quantization.quant_linear_helper import QuantLinearHelper
from tests.utils.precision_checker import PrecisionChecker

# def env
env_vars = {
    "LCCL_DETERMINISTIC": "1",
    "HCCL_DETERMINISTIC": "true",
    "ATB_MATMUL_SHUFFLE_K_ENABLE": "0",
    "ATB_LLM_LCOC_ENABLE": "0",
}


class QuantLinearRunner:
    """
    A helper class to test quantized Linear layer inference.

    This class generates dummy model configs, random inputs, float weights,
    applies quantization (if enabled), loads quantized weights, constructs
    a QuantLinear module, and finally compares the quantized output with the
    golden float output using precision checkers.
    """

    def __init__(self, dtype, quantization: str, quant_policy: str):
        """Initialize the QuantLinearRunner."""
        self.dtype = dtype
        self.quantization = quantization
        self.quant_policy = quant_policy
        self.quant_model_dir = tempfile.mkdtemp(prefix="quant_model_for_test_")
        self.fake_model_config = self._gen_model_config_from_vllm(
            self.quant_model_dir, self.quantization)
        if self.quantization == 'golden-stick':
            self.quantizer = NumpyQuantizer(self.quant_policy)

    def _gen_float_weights(self, hidden_size,
                           attention_bias) -> dict[str, np.ndarray]:
        """Generate random float32 linear weights for testing."""
        np.random.seed(42)
        weights = {}
        weight_shape = (hidden_size, hidden_size)
        weight = 0.01 * np.random.randn(*weight_shape).astype(np.float32)
        weights["linear.weight"] = weight
        bias = None
        if attention_bias:
            bias = 0.01 * np.random.randn(hidden_size).astype(np.float32)
        weights["linear.bias"] = bias
        return weights

    def _gen_input(self, hidden_size):
        """Generate random float32 input tensor for inference."""
        np.random.seed(42)
        return 0.01 * np.random.randn(3, hidden_size).astype(np.float32)

    def _create_linear(self):
        """Create a QuantLinearHelper based on vLLM config."""
        quant_config = self._gen_quant_config_from_vllm()
        return QuantLinearHelper(quant_config,
                                 self.fake_model_config.hf_config, self.dtype)

    def _load_quant_weights(self):
        """
        Load quantized weights from the safetensors file in quant_model_dir.
        """
        if not os.path.isdir(self.quant_model_dir):
            raise ValueError(
                f"Invalid quant_model_dir: {self.quant_model_dir}")
        safetensor_files = glob.glob(
            os.path.join(self.quant_model_dir, "*.safetensors"))
        if len(safetensor_files) == 1:
            safetensor_file = safetensor_files[0]
        elif len(safetensor_files) > 1:
            raise FileNotFoundError(
                f"Found multiple safetensor files in {self.quant_model_dir}")
        else:
            raise FileNotFoundError(
                f"Found no safetensor file in {self.quant_model_dir}")
        if not os.path.exists(safetensor_file):
            raise FileNotFoundError(f"File {safetensor_file} not found.")
        with safe_open(safetensor_file, framework="np", device="cpu") as f:
            weights = {}
            for key in f.keys():  # noqa: SIM118
                weights[key] = f.get_slice(key)
        return weights

    def _gen_model_config_from_vllm(self, quant_model_dir, quantization):
        """Generate a fake model config for testing from vLLM."""
        from vllm.config import ModelConfig
        json_config = {
            "architectures": ["Qwen3ForCausalLM"],
            "model_type": "qwen3",
            "hidden_size": 32,
            "attention_bias": "True",
        }
        with open(os.path.join(quant_model_dir, "config.json"),
                  "w",
                  encoding='utf-8') as f:
            json.dump(json_config, f, indent=2, ensure_ascii=False)
        print(f"config.json saved to {quant_model_dir}", flush=True)
        fake_model_config = ModelConfig(quantization=quantization,
                                        model=quant_model_dir)
        return fake_model_config

    def _gen_quant_config_from_vllm(self):
        """
        Generate vLLM quantization config if quantization is enabled.
        """
        if self.fake_model_config.quantization == "none":
            return None
        from vllm.config import LoadConfig
        from vllm.model_executor.model_loader.weight_utils import (
            get_quant_config)
        return get_quant_config(self.fake_model_config, LoadConfig())

    def run(self):
        """
        Run the full quantization + inference + precision checking pipeline.

        Steps:
            1. Generate input and float weights.
            2. If using Golden Stick, run offline quantization.
            3. Load quantized weights from safetensors.
            4. Construct QuantLinear module.
            5. Run quantized inference.
            6. Run golden float inference.
            7. Compare outputs using PrecisionChecker.
        """
        hf_config = self.fake_model_config.hf_config
        input_data = self._gen_input(hf_config.hidden_size)
        weights = self._gen_float_weights(hf_config.hidden_size,
                                          hf_config.attention_bias)
        quant_weights = {}
        if self.quantization == 'golden-stick':
            self.quantizer.quant(input_data, weights, self.quant_model_dir)
            quant_weights = self._load_quant_weights()
        linear = self._create_linear()
        linear.load_weights_into_linear(quant_weights)
        net_input = Tensor(input_data, dtype=self.dtype)
        hidden_states, _ = linear(net_input)

        golden_hidden_states = mint.nn.functional.linear(
            net_input,
            Tensor.from_numpy(weights["linear.weight"]).astype(self.dtype),
            Tensor.from_numpy(weights["linear.bias"]).astype(self.dtype))

        checker = PrecisionChecker(cos_sim_thd=0.99,
                                   l1_norm_thd=0.01,
                                   kl_dvg_thd=0.01)
        checker.check_precision(
            golden_hidden_states.astype(mstype.float32).asnumpy(),
            hidden_states.astype(mstype.float32).asnumpy())


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@patch.dict(os.environ, env_vars)
def test_gs_a8w8_bf16():
    """
    Test Summary:
        Test Golden Stick A8W8 quantization with bfloat16 dtype.
    Expected Result:
        Running successfully, precision check passed.
    Model Info:
        Linear with A8W8 quantization.
    """
    mindspore.set_device("Ascend")
    mindspore.set_deterministic(True)

    quant_runner = QuantLinearRunner(mstype.bfloat16, "golden-stick", "a8w8")
    quant_runner.run()
