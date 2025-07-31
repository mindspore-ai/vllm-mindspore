# SPDX-License-Identifier: Apache-2.0

# Adapted from
# https://github.com/vllm-project/vllm/blob/v0.8.3/vllm/utils.py
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

import contextlib
import gc
import os
import sys
from enum import Enum
from typing import TYPE_CHECKING, Generator, List, Optional, Tuple, Union

import numpy as np
import torch

if TYPE_CHECKING:
    from torch.library import Library
else:
    Library = None

import mindspore as ms
from mindspore import dtype as mstype
from mindspore.common.initializer import Zero
from mindspore._c_expression import typing
from vllm.logger import init_logger
from vllm.utils import (TORCH_DTYPE_TO_NUMPY_DTYPE, MemoryProfilingResult,
                        MemorySnapshot, T, make_ndarray_with_pad)

from .scripts import env_setup

MsKVCache = Tuple[ms.Tensor, ms.Tensor]

logger = init_logger(__name__)

STR_DTYPE_TO_MS_DTYPE = {
    "half": ms.float16,
    "float16": ms.float16,
    "bfloat16": ms.bfloat16,
    "float": ms.float32,
    "fp8": ms.uint8,
    "fp8_e4m3": ms.uint8,
    "fp8_e5m2": ms.uint8,
}

FORMAT_TYPE = {
    "nz": 29,
}

def get_valid_dtype(dtype):
    if isinstance(dtype, str):
        dtype = STR_DTYPE_TO_MS_DTYPE[dtype]
    return dtype


def get_dtype_size(dtype: torch.dtype) -> int:
    """Get the size of the data type in bytes."""
    if isinstance(dtype, str):
        dtype = STR_DTYPE_TO_TENSOR_DTYPE[dtype]
    return torch.tensor([], dtype=dtype).element_size()


def _create_empty_tensor(ms_type):
    init_func = Zero()
    init_func.__enable_zero_dim__ = True
    init_tensor = ms.Tensor(shape=(0, ), dtype=ms_type, init=init_func)
    init_tensor.init_data()

    return init_tensor


def _create_dummy_block_tables(dtype):
    return ms.ops.zeros((1, 1), dtype=dtype)


def make_tensor_with_pad(
    x: List[List[T]],
    pad: T,
    dtype: torch.dtype,
    *,
    max_len: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    pin_memory: bool = False,
) -> torch.Tensor:
    """
    Make a padded tensor from 2D inputs.

    The padding is applied to the end of each inner list until it reaches
    `max_len`.
    """
    np_dtype = TORCH_DTYPE_TO_NUMPY_DTYPE[dtype]
    padded_x = make_ndarray_with_pad(x, pad, np_dtype, max_len=max_len)

    pin_memory = False

    if padded_x.size == 0:
        tensor = _create_dummy_block_tables(dtype)
    else:
        tensor = torch.from_numpy(padded_x)
    if pin_memory:
        tensor = tensor.pin_memory()

    return tensor


def async_tensor_h2d(
    data: list,
    dtype: torch.dtype,
    target_device: Union[str, torch.device],
    pin_memory: bool,
) -> torch.Tensor:
    """Asynchronously create a tensor and copy it from host to device."""
    if not data:
        t = _create_empty_tensor(dtype)
    else:
        t = torch.tensor(data,
                         dtype=dtype,
                         pin_memory=pin_memory,
                         device="CPU")
    return t


STR_DTYPE_TO_TENSOR_DTYPE = {
    "half": torch.half,
    "float16": torch.half,
    "bfloat16": torch.bfloat16,
    "float": torch.float,
    "fp8": torch.uint8,
    "fp8_e4m3": torch.uint8,
    "fp8_e5m2": torch.uint8,
}

STR_DTYPE_TO_MS_DTYPE = {
    "half": mstype.float16,
    "float16": mstype.float16,
    "bfloat16": mstype.bfloat16,
    "float": mstype.float32,
    "fp8": mstype.uint8,
    "fp8_e4m3": mstype.uint8,
    "fp8_e5m2": mstype.uint8,
}


class vllmModelBackendEnum(str, Enum):
    """Define the variable Enum of vLLM_MODEL_BACKEND"""
    MF = 'MindFormers'
    MIND_ONE = 'MindONE'


def ascend_is_initialized():
    # Just return true for check.
    return True


def is_mindformers_model_backend():
    vllm_model_backend = os.getenv("vLLM_MODEL_BACKEND")  # noqa: SIM112
    if vllm_model_backend:
        try:
            vllmModelBackendEnum(vllm_model_backend)
            return vllm_model_backend == vllmModelBackendEnum.MF
        except ValueError as exc:
            allowed_values = [member.value for member in vllmModelBackendEnum]
            raise ValueError(
                f"Illegal value of vLLM_MODEL_BACKEND '{vllm_model_backend}',"
                f" allowed_values: {', '.join(allowed_values)}") from exc
    else:
        return False


def is_mindone_model_backend():
    return (os.getenv("vLLM_MODEL_BACKEND")  # noqa: SIM112
            and os.environ["vLLM_MODEL_BACKEND"]  # noqa: SIM112
            == vllmModelBackendEnum.MIND_ONE)


# DLLM
def register_connector():
    try:
        from vllm.distributed.kv_transfer.kv_connector.factory import (
            KVConnectorFactory)

        # use D2H for KVtransfer
        KVConnectorFactory.register_connector(
            "DLLMDsConnector", "dllm.dkvc.v1.dllm_ds_connector",
            "DLLMDsConnector")
        # use D2D for KVtransfer
        KVConnectorFactory.register_connector(
            "DLLMDsD2DConnector", "dllm.dkvc.v1.dllm_ds_d2d_connector",
            "DLLMDsD2DConnector")
    except:  # noqa: E722
        pass


def is_version_at_least(current_version, base_version):
    """
        return current_version >= base_version.
        Check whether the current version is higher than or equal to the
        base version.
        for current_version: 1.8.1, base_version: 1.11.0, it return False.
    """
    version_split_char = '.'
    if version_split_char not in base_version or version_split_char \
        not in current_version:
        raise ValueError(
            "The version string will contain the `.`."
            "For example, current_version 1.8.1， base_version: 1.11.0.")
    for x, y in zip(current_version.split(version_split_char),
                    base_version.split(version_split_char)):
        if not x.isdigit() or not y.isdigit():
            continue
        if int(x) != int(y):
            return int(x) >= int(y)
    return True


def get_ascend_soc_version():
    """Get ascend soc version."""
    if is_version_at_least(ms.__version__, "2.2.0"):
        from mindspore._c_expression import MSContext
        return MSContext.get_instance().get_ascend_soc_version()
    raise ValueError("The get_ascend_soc_version function is only "
                     "supported on MindSpore 2.6 and above.")


def is_310p():
    device = get_ascend_soc_version()
    return device in ['310p', 'ascend310p']


def convert_np_to_ms_dtype(value):
    """convert_np_to_ms_dtype"""
    if value.dtype == np.int8:
        value_dtype = ms.int8
    elif value.dtype == np.int32:
        value_dtype = ms.int32
    elif value.dtype == np.int64:
        value_dtype = ms.int64
    elif value.dtype == np.float64:
        value_dtype = ms.float64
    elif value.dtype == np.float32:
        value_dtype = ms.float32
    else:
        value_dtype = ms.bfloat16
    return value_dtype


# Replace the directly loaded module in vllm, such as 'from module import xxx'
def update_modules(name, module):
    logger.debug(f"replace module {0} by {1}".format(name, module))
    sys.modules.update({name: module})


@contextlib.contextmanager
def ms_memory_profiling(
        baseline_snapshot: MemorySnapshot,
        weights_memory: int) -> Generator[MemoryProfilingResult, None, None]:
    """Memory profiling context manager.
    baseline_snapshot: the memory snapshot before the current vLLM instance.
    weights_memory: memory used by PyTorch when loading the model weights.
        Note that, before loading the model weights, we also initialize the device
        and distributed environment, which may consume some memory. This part is not
        included in the weights_memory because PyTorch does not control it.

    The memory in one GPU can be classified into 3 categories:
    1. memory used by anything other than the current vLLM instance.
    2. memory used by torch in the current vLLM instance.
    3. memory used in the current vLLM instance, but not by torch.

    A quantitive example:

    Before creating the current vLLM instance:
        category 1: 1 GiB
        category 2: 0 GiB
        category 3: 0 GiB

    After creating the current vLLM instance and loading the model,
    (i.e. before profiling):
        category 1: 1 GiB
        category 2: 2 GiB (model weights take 2 GiB)
        category 3: 0.5 GiB (memory used by NCCL)

    During profiling (peak):
        category 1: 1 GiB
        category 2: 4 GiB (peak activation tensors take 2 GiB)
        category 3: 1 GiB (memory used by NCCL + buffers for some attention backends)

    After profiling:
        category 1: 1 GiB
        category 2: 3 GiB (after garbage-collecting activation tensors)
        category 3: 1 GiB (memory used by NCCL + buffers for some attention backends)

    In this case, non-kv cache takes 5 GiB in total, including:
    a. 2 GiB used by the model weights (category 2)
    b. 2 GiB reserved for the peak activation tensors (category 2)
    c. 1 GiB used by non-torch components (category 3)

    The memory used for loading weights (a.) is directly given from the argument `weights_memory`.

    The increase of `torch.cuda.memory_stats()["allocated_bytes.all.peak"]` during profiling gives (b.).

    The increase of `non_torch_memory` from creating the current vLLM instance until after profiling to get (c.).
    """  # noqa
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    result = MemoryProfilingResult()

    result.before_create = baseline_snapshot
    # the part of memory used for holding the model weights
    result.weights_memory = weights_memory

    result.before_profile.measure()

    yield result

    # measure memory before empty cache to get maximum reserved memory
    result.after_profile.measure()

    gc.collect()
    torch.cuda.empty_cache()

    diff_profile = result.after_profile - result.before_profile
    diff_from_create = result.after_profile - result.before_create

    # use reserved memory instead of allocated memory to describe increase of
    # torch memory
    result.torch_peak_increase = diff_profile.torch_memory
    result.non_torch_increase = diff_from_create.non_torch_memory
    result.profile_time = diff_profile.timestamp
    result.non_kv_cache_memory = result.non_torch_increase + result.torch_peak_increase + result.weights_memory  # noqa


def view(self, *shape_or_dtype):
    if len(shape_or_dtype) == 1 and isinstance(shape_or_dtype[0], typing.Type):
        target_dtype = shape_or_dtype[0]
        ori_shape = self.shape
        target_shape = (-1,)
        if len(ori_shape) > 1:
            target_shape = ori_shape[:-1] + target_shape
        out = np.frombuffer(self.numpy(), torch.ops.creation._TypeDict.get(target_dtype, np.float32))
        if not out.flags.aligned:
            out = np.require(out, requirements=["ALIGNED"])
        if target_dtype == ms.bfloat16:
            return ms.Tensor.from_numpy(out.astype(np.float32)).astype(target_dtype).reshape(target_shape)
        return ms.Tensor.from_numpy(out).reshape(target_shape)
    result = []
    if type(shape_or_dtype) is tuple:
        for items in shape_or_dtype:
            if not isinstance(items, int):
                for item in items:
                    if not isinstance(item, int):
                        result.append(item.item())
                    else:
                        result.append(item)
            else:
                result.append(items)
    return ms.ops.reshape(self, result)


def is_version_ge(current_version, base_version):
    """
        return current_version >= base_version.
        Check whether the current version is higher than or equal to the base version.
        for current_version: 1.8.1, base_version: 1.11.0, it return False.
    """
    version_split_char = '.'
    if version_split_char not in base_version or version_split_char not in current_version:
        raise ValueError("The version string will contain the `.`."
                         "For example, current_version 1.8.1， base_version: 1.11.0.")
    for x, y in zip(current_version.split(version_split_char), base_version.split(version_split_char)):
        if not x.isdigit() or not y.isdigit():
            continue
        if int(x) != int(y):
            return int(x) >= int(y)
    return True

def get_ascend_soc_version():
    """Get ascend soc version."""
    if is_version_ge(ms.__version__, "2.2.0"):
        from mindspore._c_expression import MSContext
        return MSContext.get_instance().get_ascend_soc_version()
    ascend_chip_type = os.getenv("ASCEND_CHIP_TYPE", "UNSET")
    if ascend_chip_type not in ["910a", "910b", "UNSET"]:
        raise EnvironmentError(f"ASCEND_CHIP_TYPE should be in ['910a', '910b'],but get {ascend_chip_type}")
    if ascend_chip_type == "UNSET":
        logger.info("Environment variables need to be set manually to obtain the chip type,"
                    "which can be set as follows: \n"
                    "For Atlas 800, run 'export ASCEND_CHIP_TYPE=910a' before the program runs.\n"
                    "For Atlas 800T A2, run 'export ASCEND_CHIP_TYPE=910b' before the program runs.\n"
                    "If you need to get chip information automatically, MindSpore 2.2 and above is recommended")
    return ascend_chip_type

def atlas_inference():
    device = get_ascend_soc_version()
    return device in ['310p', 'ascend310p']

def check_ready():
    from mindspore import set_context

    # Common environment variables of predict.
    set_context(jit_config={"jit_level": "O0", "infer_boost": "on"})
    default_env = {
        "MS_INTERNAL_DISABLE_CUSTOM_KERNEL_LIST":
        "FlashAttentionScore,PagedAttention",
    }
    if atlas_inference():
        default_env["MS_ENABLE_INTERNAL_BOOST"] = "off"
    env_setup(default_env)

    if os.getenv("MS_MEMPOOL_BLOCK_SIZE"):
        set_context(
            mempool_block_size=f"{os.environ['MS_MEMPOOL_BLOCK_SIZE']}GB")

    if is_mindformers_model_backend():
        logger.info("Run with Mindformers backend!")
    elif is_mindone_model_backend():
        logger.info("Run with MindONE backend!")
    else:
        logger.info("Run with native model backend!")
    register_connector()
