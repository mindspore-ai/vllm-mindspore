import pytest
import torch

from tests.mark_utils import arg_mark
from tests.ops_utils import AssertRtolEqual
from mrt.torch import backend


def op_func(input_tensor, chunks, dim=0):
  return input_tensor.chunk(chunks, dim)


def get_op_func_compiled():
  def custom_op_func(input_tensor, chunks, dim=0):
    return input_tensor.chunk(chunks, dim)

  return torch.compile(custom_op_func, backend=backend)


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("pipeline", (True, False))
@pytest.mark.parametrize("shape, dim, chunks", [
  ([128, 4096], 0, 2),
  ([33, 1024], 0, 4),   # dim size not divisible by chunks
  ([16, 33], 1, 4),     # split along non-leading dim, not divisible
])
def test_chunk(pipeline, monkeypatch, shape, dim, chunks):
  """
  Feature: Test torch.chunk
  Description: Compare compiled NPU chunk results with CPU reference
  Expectation: All chunk outputs match between CPU and NPU
  """
  if pipeline:
    monkeypatch.setenv("MRT_ENABLE_PIPELINE", "on")

  cpu_input = torch.rand(shape, dtype=torch.bfloat16)
  npu_input = cpu_input.npu()

  cpu_chunks = op_func(cpu_input, chunks, dim=dim)
  op_func_compiled = get_op_func_compiled()
  npu_chunks = op_func_compiled(npu_input, chunks, dim=dim)

  assert len(cpu_chunks) == len(npu_chunks)
  for cpu_out, npu_out in zip(cpu_chunks, npu_chunks):
    AssertRtolEqual(cpu_out, npu_out.detach().cpu())
