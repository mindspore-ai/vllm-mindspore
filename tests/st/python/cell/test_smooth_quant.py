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
"""test_smooth_quant"""
import mindspore
import numpy as np
import pytest
from mindspore import Parameter, Tensor, context
from mindspore import dtype as mstype
from mindspore import nn
from mindspore.common.initializer import initializer

from vllm_mindspore.model_executor.layers.quantization.smooth_quant_modelslim \
    import A8W8DYNLinearMethod, A8W8LinearMethod, SmoothQuantModelSlimConfig


@pytest.fixture(params=[mstype.float16, mstype.bfloat16],
                ids=["float16", "bfloat16"])
def params_dtype(request):
    return request.param


@pytest.fixture(params=[True, False], ids=["grouped", "no-grouped"])
def is_group_mm(request):
    return request.param


@pytest.fixture
def quant_config(is_group_mm):
    if is_group_mm:
        quant_config = SmoothQuantModelSlimConfig({},
                                                  weight_bits=8,
                                                  group_size=4)
    else:
        quant_config = SmoothQuantModelSlimConfig({},
                                                  weight_bits=8,
                                                  group_size=1)
    return quant_config


class SimpleLinear(nn.Cell):

    def __init__(self, method, params_dtype):
        super().__init__()
        self.method = method
        self.params_dtype = params_dtype

    def construct(self,
                  x: mindspore.Tensor,
                  bias: mindspore.Parameter = None,
                  group_list=None,
                  cumsum_flag=False):
        return self.method.apply(self, x, bias, group_list, cumsum_flag)


class LinearMethodTestBase:

    def apply_test_template(self, method, params_dtype, is_group_mm):
        context.set_context(jit_config={
            "jit_level": "O0",
            "infer_boost": "on"
        })
        context.set_context(mode=context.GRAPH_MODE)
        simple_layer = SimpleLinear(method, params_dtype)
        input_size = 5120
        output_size = 1344
        expert_num = 8 if is_group_mm else 1

        method.create_weights(layer=simple_layer,
                              input_size_per_partition=input_size,
                              output_partition_sizes=[output_size],
                              input_size=input_size,
                              output_size=output_size,
                              params_dtype=simple_layer.params_dtype,
                              is_group_mm=is_group_mm,
                              expert_num_per_partition=expert_num)
        method.process_weights_after_loading(simple_layer)

        bias = Parameter(initializer('zeros', (output_size, ),
                                     simple_layer.params_dtype),
                         requires_grad=False)
        group_list = None
        if is_group_mm:
            group_list = Tensor(np.array([0, 0, 1, 1, 1, 2, 0, 0]),
                                dtype=mindspore.int64)

        batch_size = 1
        seq_len = 5
        x_shape = (batch_size, seq_len, input_size)
        input_tensor = Tensor(np.random.rand(*x_shape), dtype=params_dtype)
        output = simple_layer(x=input_tensor, bias=bias, group_list=group_list)

        assert output.shape == (batch_size, seq_len, output_size)


class TestA8W8LinearMethod(LinearMethodTestBase):

    def test_create_weights(self, quant_config, params_dtype, is_group_mm):
        """Test static quant create weight methods."""
        method = A8W8LinearMethod(quant_config)
        simple_layer = SimpleLinear(method, params_dtype)
        input_size = 5120
        output_size = 1344
        expert_num = 8 if is_group_mm else 1

        method.create_weights(layer=simple_layer,
                              input_size_per_partition=input_size,
                              output_partition_sizes=[output_size],
                              input_size=input_size,
                              output_size=output_size,
                              params_dtype=simple_layer.params_dtype,
                              is_group_mm=is_group_mm,
                              expert_num_per_partition=expert_num)

        # verify the existence of the weight parameter
        assert hasattr(simple_layer, 'weight')
        assert hasattr(simple_layer, 'weight_scale')
        assert hasattr(simple_layer, 'deq_scale')
        assert hasattr(simple_layer, 'input_scale')
        assert hasattr(simple_layer, 'input_offset')

        # verify the dtype
        expected_scale_dtype = mstype.bfloat16 if simple_layer.params_dtype \
                                == mstype.bfloat16 else mstype.float32
        assert simple_layer.weight_scale.dtype == expected_scale_dtype

        assert simple_layer.deq_scale.dtype == mstype.int64

        assert simple_layer.input_scale.dtype == simple_layer.params_dtype
        assert simple_layer.input_offset.dtype == simple_layer.params_dtype

    def test_apply(self, quant_config, params_dtype, is_group_mm):
        """Test static quant apply method."""
        method = A8W8LinearMethod(quant_config)
        self.apply_test_template(method, params_dtype, is_group_mm)

    def test_process_weights_after_loading(self, quant_config, params_dtype,
                                           is_group_mm):
        """Test static quant process weights after loading."""
        method = A8W8LinearMethod(quant_config)
        simple_layer = SimpleLinear(method, params_dtype)
        input_size = 5120
        output_size = 1344
        expert_num = 8 if is_group_mm else 1

        method.create_weights(layer=simple_layer,
                              input_size_per_partition=input_size,
                              output_partition_sizes=[output_size],
                              input_size=input_size,
                              output_size=output_size,
                              params_dtype=simple_layer.params_dtype,
                              is_group_mm=is_group_mm,
                              expert_num_per_partition=expert_num)

        method.process_weights_after_loading(simple_layer)
        # weight shape verification
        if is_group_mm:
            assert simple_layer.weight.shape == (expert_num, input_size,
                                                 output_size)
            assert simple_layer.weight_scale.shape == (expert_num, output_size)
            assert simple_layer.deq_scale.shape == (expert_num, output_size)
            assert simple_layer.input_scale.shape == (1, )
            assert simple_layer.input_offset.shape == (1, )
        else:
            assert simple_layer.weight.shape == (output_size, input_size)
            assert simple_layer.weight_scale.shape == (output_size, 1)
            assert simple_layer.deq_scale.shape == (output_size, )
            assert simple_layer.input_scale.shape == (1, )
            assert simple_layer.input_offset.shape == (1, )

        assert isinstance(simple_layer.input_offset, Parameter)
        assert simple_layer.input_offset.dtype == mstype.int8

        # special processing verify for bfloat16
        if simple_layer.params_dtype == mstype.bfloat16 and \
            not method.is_310p:
            assert simple_layer.deq_scale.dtype == mstype.float32


class TestA8W8DYNLinearMethod(LinearMethodTestBase):

    def test_create_weights(self, quant_config, params_dtype, is_group_mm):
        """Test dynamic quant create weight methods."""
        method = A8W8DYNLinearMethod(quant_config)
        simple_layer = SimpleLinear(method, params_dtype)
        input_size = 5120
        output_size = 1344
        expert_num = 8 if is_group_mm else 1

        method.create_weights(layer=simple_layer,
                              input_size_per_partition=input_size,
                              output_partition_sizes=[output_size],
                              input_size=input_size,
                              output_size=output_size,
                              params_dtype=simple_layer.params_dtype,
                              is_group_mm=is_group_mm,
                              expert_num_per_partition=expert_num,
                              is_2d_smooth_scale=True)

        # verify the existence of the weight parameter
        assert hasattr(simple_layer, 'weight')
        assert hasattr(simple_layer, 'weight_scale')
        assert hasattr(simple_layer, 'smooth_scale')

        # verify the dtype of smooth_scale
        assert simple_layer.smooth_scale.dtype == simple_layer.params_dtype

    def test_apply(self, quant_config, params_dtype, is_group_mm):
        """Test dynamic quant apply methods."""
        method = A8W8DYNLinearMethod(quant_config)
        self.apply_test_template(method, params_dtype, is_group_mm)

    def test_process_weights_after_loading(self, quant_config, params_dtype,
                                           is_group_mm):
        """Test dynamic quant process weights after loading."""
        method = A8W8DYNLinearMethod(quant_config)
        simple_layer = SimpleLinear(method, params_dtype)
        input_size = 5120
        output_size = 1344
        expert_num = 8 if is_group_mm else 1

        method.create_weights(layer=simple_layer,
                              input_size_per_partition=input_size,
                              output_partition_sizes=[output_size],
                              input_size=input_size,
                              output_size=output_size,
                              params_dtype=simple_layer.params_dtype,
                              is_group_mm=is_group_mm,
                              expert_num_per_partition=expert_num,
                              is_2d_smooth_scale=True)

        method.process_weights_after_loading(simple_layer)
        if is_group_mm:
            assert simple_layer.weight.shape == (expert_num, input_size,
                                                 output_size)
            assert simple_layer.weight_scale.shape == (expert_num, output_size)
        else:
            assert simple_layer.weight.shape == (output_size, input_size)
            assert simple_layer.weight_scale.shape == (output_size, )

        assert simple_layer.smooth_scale.shape == (1, input_size)
        assert simple_layer.smooth_scale.dtype == mstype.float32


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_linear_method(quant_config, params_dtype, is_group_mm):
    a8w8_test = TestA8W8LinearMethod()
    a8w8_test.test_create_weights(quant_config, params_dtype, is_group_mm)
    a8w8_test.test_apply(quant_config, params_dtype, is_group_mm)
    a8w8_test.test_process_weights_after_loading(quant_config, params_dtype,
                                                 is_group_mm)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_dyn_linear_method(quant_config, params_dtype, is_group_mm):
    a8w8_dyn_test = TestA8W8DYNLinearMethod()
    a8w8_dyn_test.test_create_weights(quant_config, params_dtype, is_group_mm)
    a8w8_dyn_test.test_apply(quant_config, params_dtype, is_group_mm)
    a8w8_dyn_test.test_process_weights_after_loading(quant_config,
                                                     params_dtype, is_group_mm)
