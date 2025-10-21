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

""" Utility functions for testing operations """

import torch
import numpy as np
from typing import Sequence
from numbers import Number


def AssertRtolEqual(x, y, prec=1.e-4, prec16=1.e-3, auto_trans_dtype=False, message=None):
    """
    Assert that two tensors are equal within a relative tolerance
    Args:
        x: The first tensor.
        y: The second tensor.
        prec: The relative tolerance.
        prec16: The relative tolerance for 16-bit floats.
        auto_trans_dtype: Whether to automatically transpose the data type.
        message: The message to display if the assertion fails.
    """

    def _assertRtolEqual(x, y, prec, prec16, message):
        def compare_res(pre, minimum):
            diff = y - x
            # check that NaNs are in the same locations
            nan_mask = np.isnan(x)
            if not np.equal(nan_mask, np.isnan(y)).all():
                raise AssertionError("NaN locations do not match")
            if nan_mask.any():
                diff[nan_mask] = 0
            result = np.abs(diff)
            deno = np.maximum(np.abs(x), np.abs(y))
            result_atol = np.less_equal(result, pre)
            result_rtol = np.less_equal(result / np.add(deno, minimum), pre)
            if not result_rtol.all() and not result_atol.all():
                if np.sum(result_rtol == False) > size * pre and np.sum(result_atol == False) > size * pre:
                    raise AssertionError("result error")

        minimum16 = 6e-8
        minimum = 10e-10

        if isinstance(x, Sequence) and isinstance(y, Sequence):
            for x_, y_ in zip(x, y):
                _assertRtolEqual(x_, y_, prec, prec16, message)
            return

        if isinstance(x, torch.Tensor) and isinstance(y, Sequence):
            y = torch.as_tensor(y, dtype=x.dtype, device=x.device)
        elif isinstance(x, Sequence) and isinstance(y, torch.Tensor):
            x = torch.as_tensor(x, dtype=y.dtype, device=y.device)

        if torch.is_tensor(x) and torch.is_tensor(y):
            if auto_trans_dtype:
                x = x.to(y.dtype)
            if (x.dtype == torch.bfloat16) and (y.dtype == torch.bfloat16):
                if x.shape != y.shape:
                    raise AssertionError("shape error!")
                result = torch.allclose(x.cpu(), y.cpu(), rtol=prec16, atol=prec16)
                if not result:
                    raise AssertionError("result error!")
                return
            x = x.detach().cpu().numpy()
            y = y.detach().cpu().numpy()
        elif isinstance(x, Number) and isinstance(y, Number):
            x = np.array(x)
            y = np.array(y)

        size = x.size
        if x.shape != y.shape:
            raise AssertionError("shape error")
        if x.dtype != y.dtype:
            raise AssertionError("dtype error")
        dtype_list = [np.bool_, np.uint16, np.int16, np.int32, np.float16, 
                    np.float32, np.int8, np.uint8, np.int64, np.float64]
        if x.dtype not in dtype_list:
            raise AssertionError("required dtype in [np.bool_, np.uint16, np.int16, " +
                    "np.int32, np.float16, np.float32, np.int8, np.uint8, np.int64]")
        if x.dtype == np.bool_:
            result = np.equal(x, y)
            if not result.all():
                raise AssertionError("result error")
        elif x.dtype == np.float16:
            compare_res(prec16, minimum16)
        elif x.dtype in [np.float32, np.int8, np.uint8, np.uint16, np.int16, np.int32, np.int64, np.float64]:
            compare_res(prec, minimum)
        else:
            raise AssertionError("required dtype must be numpy object")

    _assertRtolEqual(x, y, prec, prec16, message)
