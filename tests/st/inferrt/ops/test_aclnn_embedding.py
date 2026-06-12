"""Tests for aclnn embedding operation."""
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from ms_inferrt.torch import fx_mlir_backend as backend

from tests.mark_utils import arg_mark
from tests.ops_utils import AssertRtolEqual


def op_func(indices, weight):
    out = F.embedding(indices, weight)
    return out


def get_op_func_compiled():
    def custom_op_func(indices, weight):
        return F.embedding(indices, weight)
    return torch.compile(custom_op_func, backend=backend)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("dtype", (torch.float32, torch.float16))
def test_embedding(dtype):
    """
    Feature: Test aclnn embedding
    Description: Test aclnn embedding with fp32/fp16 inputs
    Expectation: The result is correct
    """
    cpu_weight_torch = torch.from_numpy(np.random.rand(10, 3).astype(np.float32)).to(dtype)
    cpu_indices = torch.from_numpy(np.array([[1, 2, 4, 5], [4, 3, 2, 9]]))

    npu_weight_torch = cpu_weight_torch.npu()
    npu_indices = cpu_indices.npu()

    cpu_output0 = op_func(cpu_indices, cpu_weight_torch)
    op_func_compiled = get_op_func_compiled()
    npu_output0 = [npu_output.detach().cpu().numpy() for npu_output in op_func_compiled(npu_indices, npu_weight_torch)]
    AssertRtolEqual(cpu_output0, npu_output0)

    cpu_weight_torch = torch.from_numpy(np.random.rand(20, 4).astype(np.float32)).to(dtype)
    cpu_indices = torch.from_numpy(np.array([[1, 2, 4, 5], [4, 3, 2, 9]]))

    npu_weight_torch = cpu_weight_torch.npu()
    npu_indices = cpu_indices.npu()

    cpu_output0 = op_func(cpu_indices, cpu_weight_torch)
    op_func_compiled = get_op_func_compiled()
    npu_output0 = [npu_output.detach().cpu().numpy() for npu_output in op_func_compiled(npu_indices, npu_weight_torch)]
    AssertRtolEqual(cpu_output0, npu_output0)

def aten_embedding_dynamic_op(weight, indices):
    num_words = indices.size(1)
    return torch.ops.aten.embedding.default(weight, indices[:, :num_words])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("vocab_size,embed_dim,num_words",
                         [(10, 4, 3), (20, 8, 4), (32, 16, 5),
                          (64, 32, 8), (128, 64, 4)])
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_aten_embedding_dynamic(vocab_size, embed_dim, num_words, dtype):
    """
Feature: Test aten embedding with dynamic shapes.
    Description: Test aten.embedding.default with various vocab sizes.
    Expectation: The result matches eager mode.
    """
    compiled_op = torch.compile(aten_embedding_dynamic_op, backend=backend)
    prec = 0.001 if dtype == np.float16 else 0.0001
    cpu_weight = np.random.rand(vocab_size, embed_dim).astype(dtype)
    cpu_indices = np.random.randint(0, vocab_size, (2, num_words))
    npu_weight = torch.from_numpy(cpu_weight).npu()
    npu_indices = torch.from_numpy(cpu_indices).npu()
    cpu_output = aten_embedding_dynamic_op(
        torch.from_numpy(cpu_weight), torch.from_numpy(cpu_indices)
    ).detach().numpy()
    npu_output = compiled_op(
        npu_weight, npu_indices
    ).detach().cpu().numpy()
    AssertRtolEqual(cpu_output, npu_output, prec)

def embedding_dynamic_op(indices, weight):
    num_words = indices.size(1)
    return F.embedding(indices[:, :num_words], weight)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("vocab_size,embed_dim,num_words",
                         [(10, 4, 3), (20, 8, 4), (32, 16, 5),
                          (64, 32, 8), (128, 64, 4)])
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_embedding_dynamic(vocab_size, embed_dim, num_words, dtype):
    """
Feature: Test embedding with dynamic shapes.
    Description: Test F.embedding with dynamic word count slicing.
    Expectation: The result matches eager mode.
    """
    compiled_op = torch.compile(embedding_dynamic_op, backend=backend)
    prec = 0.001 if dtype == np.float16 else 0.0001
    cpu_weight = np.random.rand(vocab_size, embed_dim).astype(dtype)
    cpu_indices = np.random.randint(0, vocab_size, (2, num_words))
    npu_weight = torch.from_numpy(cpu_weight).npu()
    npu_indices = torch.from_numpy(cpu_indices).npu()
    cpu_output = embedding_dynamic_op(
        torch.from_numpy(cpu_indices), torch.from_numpy(cpu_weight)
    ).detach().numpy()
    npu_output = compiled_op(
        npu_indices, npu_weight
    ).detach().cpu().numpy()
    AssertRtolEqual(cpu_output, npu_output, prec)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("vocab_size,embed_dim", [(10, 4), (20, 8), (64, 32)])
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_embedding_static(vocab_size, embed_dim, dtype):
    """
Feature: Test embedding with static shapes.
    Description: Test F.embedding with fixed vocab and embed dimensions.
    Expectation: The result matches eager mode.
    """
    def embedding_static_op(indices, weight):
        return F.embedding(indices, weight)

    compiled_op = torch.compile(embedding_static_op, backend=backend)
    prec = 0.001 if dtype == np.float16 else 0.0001
    num_words = 4
    cpu_weight = np.random.rand(vocab_size, embed_dim).astype(dtype)
    cpu_indices = np.random.randint(0, vocab_size, (2, num_words))
    npu_weight = torch.from_numpy(cpu_weight).npu()
    npu_indices = torch.from_numpy(cpu_indices).npu()
    cpu_output = embedding_static_op(torch.from_numpy(cpu_indices), torch.from_numpy(cpu_weight)).detach().numpy()
    npu_output = compiled_op(
        npu_indices, npu_weight
    ).detach().cpu().numpy()
    AssertRtolEqual(cpu_output, npu_output, prec)
