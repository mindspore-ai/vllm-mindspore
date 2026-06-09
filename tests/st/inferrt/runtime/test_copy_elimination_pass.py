"""Unit tests for the copy_elimination FX graph pass."""

import operator
import importlib.util
from pathlib import Path

import torch
from torch.fx import Graph, GraphModule


_COPY_ELIMINATION_PATH = (
    Path(__file__).resolve().parents[4]
    / "inferrt"
    / "python"
    / "ms_inferrt"
    / "torch"
    / "copy_elimination.py"
)
_SPEC = importlib.util.spec_from_file_location("copy_elimination_under_test", _COPY_ELIMINATION_PATH)
_COPY_ELIMINATION = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_COPY_ELIMINATION)
eliminate_redundant_copy_ = _COPY_ELIMINATION.eliminate_redundant_copy_


def _copy_nodes(gm):
    return [node for node in gm.graph.nodes if node.op == "call_method" and node.target == "copy_"]


def test_eliminate_copy_when_destination_is_linear():
    """
    Feature: Copy elimination pass
    Description: Eliminate copy_ when destination is a linear output and redirect users to source
    Expectation: copy_ node is removed and downstream user consumes the source tensor
    """
    graph = Graph()
    x = graph.placeholder("x")
    weight = graph.placeholder("weight")
    src = graph.placeholder("src")
    linear = graph.call_function(torch.nn.functional.linear, (x, weight))
    graph.call_method("copy_", (linear, src))
    user = graph.call_function(operator.neg, (linear,))
    graph.output(user)
    gm = GraphModule({}, graph)

    eliminate_redundant_copy_(gm)

    assert _copy_nodes(gm) == []
    assert user.args == (src,)


def test_keep_copy_when_destination_is_not_linear():
    """
    Feature: Copy elimination pass
    Description: Keep copy_ when destination is not produced by linear
    Expectation: copy_ node remains and downstream user still consumes the original destination
    """
    graph = Graph()
    x = graph.placeholder("x")
    y = graph.placeholder("y")
    src = graph.placeholder("src")
    add = graph.call_function(operator.add, (x, y))
    graph.call_method("copy_", (add, src))
    user = graph.call_function(operator.neg, (add,))
    graph.output(user)
    gm = GraphModule({}, graph)

    eliminate_redundant_copy_(gm)

    assert len(_copy_nodes(gm)) == 1
    assert user.args == (add,)
