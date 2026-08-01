"""Unit tests for the no-op cast elimination FX graph pass."""

import importlib.util
from pathlib import Path

import pytest
import torch
from torch.fx import Graph, GraphModule


_CAST_ELIMINATION_PATH = (
    Path(__file__).resolve().parents[4]
    / "inferrt"
    / "python"
    / "ms_inferrt"
    / "torch"
    / "cast_elimination.py"
)
_SPEC = importlib.util.spec_from_file_location("cast_elimination_under_test", _CAST_ELIMINATION_PATH)
_CAST_ELIMINATION = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_CAST_ELIMINATION)
eliminate_noop_casts_ = _CAST_ELIMINATION.eliminate_noop_casts_
_DISABLE_CAST_ELIMINATION_ENV = "MS_INFERRT_DISABLE_CAST_ELIMINATION"


def _tensor_meta(dtype=torch.float32, device="cpu"):
    return torch.empty((2, 3), dtype=dtype, device=device)


def _set_meta(node, dtype=torch.float32, device="cpu"):
    node.meta["example_value"] = _tensor_meta(dtype=dtype, device=device)
    return node


def _build_graph_with_cast(cast_builder, input_dtype=torch.float32, output_dtype=torch.float32):
    graph = Graph()
    x = graph.placeholder("x")
    _set_meta(x, input_dtype)
    cast = cast_builder(graph, x)
    _set_meta(cast, output_dtype)
    user = graph.call_function(torch.ops.aten.neg.default, (cast,))
    _set_meta(user, output_dtype)
    graph.output(user)
    return GraphModule({}, graph), cast, user, x


def _cast_nodes(gm):
    targets = (
        {"to"}
        | set(_CAST_ELIMINATION._DTYPE_METHOD_TARGETS.keys())  # pylint: disable=protected-access
        | _CAST_ELIMINATION._CAST_FUNCTION_TARGETS  # pylint: disable=protected-access
    )
    return [node for node in gm.graph.nodes if node.target in targets]


_DTYPE_METHOD_CASES = [
    (f"method_{target}", lambda graph, x, target=target: graph.call_method(target, (x,)), dtype, dtype)
    for target, dtype in _CAST_ELIMINATION._DTYPE_METHOD_TARGETS.items()  # pylint: disable=protected-access
]


@pytest.mark.parametrize(
    "target_name,cast_builder,input_dtype,output_dtype",
    [
        (
            "method_to_positional_dtype",
            lambda graph, x: graph.call_method("to", (x, torch.float32)),
            torch.float32,
            torch.float32,
        ),
        (
            "method_to_keyword_dtype",
            lambda graph, x: graph.call_method("to", (x,), {"dtype": torch.float32}),
            torch.float32,
            torch.float32,
        ),
        (
            "method_to_device_dtype_args",
            lambda graph, x: graph.call_method("to", (x, torch.device("cpu"), torch.float32)),
            torch.float32,
            torch.float32,
        ),
        (
            "method_to_string_device_dtype_args",
            lambda graph, x: graph.call_method("to", (x, "cpu", torch.float32)),
            torch.float32,
            torch.float32,
        ),
        (
            "method_to_tensor_arg",
            lambda graph, x: graph.call_method(
                "to", (x, _set_meta(graph.placeholder("y"), torch.float32))
            ),
            torch.float32,
            torch.float32,
        ),
        (
            "prims_convert_element_type",
            lambda graph, x: graph.call_function(torch.ops.prims.convert_element_type.default, (x, torch.float32)),
            torch.float32,
            torch.float32,
        ),
        (
            "aten_to_dtype",
            lambda graph, x: graph.call_function(torch.ops.aten.to.dtype, (x, torch.float32, False, False, None)),
            torch.float32,
            torch.float32,
        ),
        (
            "aten_to_device",
            lambda graph, x: graph.call_function(
                torch.ops.aten.to.device,
                (x, torch.device("cpu"), torch.float32, False, False, None),
            ),
            torch.float32,
            torch.float32,
        ),
        (
            "aten_to_other",
            lambda graph, x: graph.call_function(
                torch.ops.aten.to.other,
                (x, _set_meta(graph.placeholder("other"), torch.float32), False, False, None),
            ),
            torch.float32,
            torch.float32,
        ),
        (
            "aten_to_dtype_layout",
            lambda graph, x: graph.call_function(
                torch.ops.aten.to.dtype_layout,
                (x,),
                {
                    "dtype": torch.float32,
                    "layout": torch.strided,
                    "device": torch.device("cpu"),
                    "pin_memory": None,
                    "non_blocking": False,
                    "copy": False,
                    "memory_format": None,
                },
            ),
            torch.float32,
            torch.float32,
        ),
        (
            "aten_to_prim_device",
            lambda graph, x: graph.call_function(
                torch.ops.aten.to.prim_Device,
                (x, torch.device("cpu"), None, False, False),
            ),
            torch.float32,
            torch.float32,
        ),
        (
            "aten_to_prim_dtype",
            lambda graph, x: graph.call_function(
                torch.ops.aten.to.prim_dtype,
                (x, 6, False, False),
            ),
            torch.float32,
            torch.float32,
        ),
        (
            "aten_to_prim_other",
            lambda graph, x: graph.call_function(
                torch.ops.aten.to.prim_other,
                (x, False, False),
            ),
            torch.float32,
            torch.float32,
        ),
        *_DTYPE_METHOD_CASES,
    ],
)
def test_eliminate_noop_cast_targets(target_name, cast_builder, input_dtype, output_dtype):
    """
    Feature: No-op cast elimination pass
    Description: Remove every supported FX target when input/output dtype matches
    Expectation: Cast node is removed and downstream user consumes the original input
    """
    gm, cast, user, x = _build_graph_with_cast(cast_builder, input_dtype, output_dtype)

    eliminate_noop_casts_(gm)

    assert cast not in gm.graph.nodes, target_name
    assert user.args == (x,)
    assert _cast_nodes(gm) == []


@pytest.mark.parametrize(
    "device_arg,target_dtype",
    [
        ("cpu", torch.float16),
        ("cpu:0", torch.bfloat16),
        ("npu", torch.float32),
        (torch.device("cpu"), torch.float64),
    ],
)
def test_parse_to_method_dtype_after_device(device_arg, target_dtype):
    """
    Feature: Tensor.to argument parsing
    Description: Parse dtype after string and torch.device arguments
    Expectation: The third positional argument is recognized as the requested dtype
    """
    graph = Graph()
    x = _set_meta(graph.placeholder("x"), torch.float32)
    cast = graph.call_method("to", (x, device_arg, target_dtype))

    parser = _CAST_ELIMINATION._target_dtype_for_to_method  # pylint: disable=W0212
    parsed_dtype = parser(cast)

    assert parsed_dtype == target_dtype


def test_eliminate_noop_cast_used_as_graph_output(monkeypatch):
    """
    Feature: No-op cast elimination pass
    Description: Eliminate a no-op cast directly referenced by the output node
    Expectation: The graph returns the original input without a cast node
    """
    monkeypatch.delenv(_DISABLE_CAST_ELIMINATION_ENV, raising=False)
    graph = Graph()
    x = graph.placeholder("x")
    _set_meta(x, torch.float32)
    cast = graph.call_method("to", (x, torch.float32))
    _set_meta(cast, torch.float32)
    output = graph.output(cast)
    gm = GraphModule({}, graph)

    eliminate_noop_casts_(gm)

    assert cast not in gm.graph.nodes
    assert output.args == (x,)
    assert _cast_nodes(gm) == []
    gm.graph.lint()

    input_tensor = torch.randn((2, 3), dtype=torch.float32)
    assert gm(input_tensor) is input_tensor


def test_cast_elimination_is_enabled_by_default(monkeypatch):
    """
    Feature: No-op cast elimination switch
    Description: Run the pass without configuring its disable environment variable
    Expectation: Redundant casts are eliminated by default
    """
    monkeypatch.delenv(_DISABLE_CAST_ELIMINATION_ENV, raising=False)
    gm, cast, user, x = _build_graph_with_cast(
        lambda graph, input_node: graph.call_method(
            "to", (input_node, torch.float32)
        ),
        input_dtype=torch.float32,
        output_dtype=torch.float32,
    )

    eliminate_noop_casts_(gm)

    assert cast not in gm.graph.nodes
    assert user.args == (x,)


def test_cast_elimination_can_be_disabled(monkeypatch):
    """
    Feature: No-op cast elimination switch
    Description: Disable the pass through its environment variable
    Expectation: Redundant casts remain in the FX graph
    """
    monkeypatch.setenv(_DISABLE_CAST_ELIMINATION_ENV, "1")
    gm, cast, user, _ = _build_graph_with_cast(
        lambda graph, input_node: graph.call_method(
            "to", (input_node, torch.float32)
        ),
        input_dtype=torch.float32,
        output_dtype=torch.float32,
    )

    eliminate_noop_casts_(gm)

    assert cast in gm.graph.nodes
    assert user.args == (cast,)


def test_keep_aten_to_copy_when_dtype_is_unchanged():
    """
    Feature: No-op cast elimination pass
    Description: Preserve aten._to_copy because it retains copy semantics
    Expectation: aten._to_copy remains after cast elimination
    """
    gm, to_copy, user, _ = _build_graph_with_cast(
        lambda graph, x: graph.call_function(
            torch.ops.aten._to_copy.default,  # pylint: disable=protected-access
            (x,),
            {"dtype": torch.float32},
        ),
        input_dtype=torch.float32,
        output_dtype=torch.float32,
    )

    eliminate_noop_casts_(gm)

    assert to_copy in gm.graph.nodes
    assert user.args == (to_copy,)


def test_aten_to_copy_preserves_storage_and_mutation_semantics():
    """
    Feature: No-op cast elimination pass
    Description: Execute aten._to_copy after the pass
    Expectation: The result has independent storage and mutations do not affect the input
    """
    graph = Graph()
    x = _set_meta(graph.placeholder("x"), torch.float32)
    to_copy = graph.call_function(
        torch.ops.aten._to_copy.default,  # pylint: disable=protected-access
        (x,),
        {"dtype": torch.float32},
    )
    _set_meta(to_copy, torch.float32)
    graph.output(to_copy)
    gm = GraphModule({}, graph)

    eliminate_noop_casts_(gm)

    input_tensor = torch.randn((2, 3), dtype=torch.float32)
    original_input = input_tensor.clone()
    output = gm(input_tensor)

    torch.testing.assert_close(output, input_tensor)
    assert output.untyped_storage().data_ptr() != input_tensor.untyped_storage().data_ptr()

    output.add_(1)
    torch.testing.assert_close(input_tensor, original_input)


def test_keep_cast_when_dtype_changes():
    """
    Feature: No-op cast elimination pass
    Description: Keep real dtype conversion
    Expectation: Cast node remains
    """
    gm, cast, user, _ = _build_graph_with_cast(
        lambda graph, x: graph.call_method("to", (x, torch.float16)),
        input_dtype=torch.float32,
        output_dtype=torch.float16,
    )

    eliminate_noop_casts_(gm)

    assert cast in gm.graph.nodes
    assert user.args == (cast,)


@pytest.mark.parametrize("device_arg", ["cpu", "cpu:0"])
@pytest.mark.parametrize(
    "output_metadata_dtype",
    [torch.float16, torch.float32],
    ids=["accurate_metadata", "stale_metadata"],
)
def test_keep_to_with_string_device_when_requested_dtype_changes(
    device_arg, output_metadata_dtype
):
    """
    Feature: No-op cast elimination pass
    Description: Preserve dtype conversion requested after a string device argument
    Expectation: Explicit dtype prevents elimination even with stale output metadata
    """
    gm, cast, user, _ = _build_graph_with_cast(
        lambda graph, x: graph.call_method(
            "to", (x, device_arg, torch.float16)
        ),
        input_dtype=torch.float32,
        output_dtype=output_metadata_dtype,
    )

    eliminate_noop_casts_(gm)

    assert cast in gm.graph.nodes
    assert user.args == (cast,)


def test_keep_to_when_copy_true():
    """
    Feature: No-op cast elimination pass
    Description: Preserve explicit copy=True semantics
    Expectation: to node remains even if dtype does not change
    """
    gm, cast, user, _ = _build_graph_with_cast(
        lambda graph, x: graph.call_method("to", (x,), {"dtype": torch.float32, "copy": True}),
        input_dtype=torch.float32,
        output_dtype=torch.float32,
    )

    eliminate_noop_casts_(gm)

    assert cast in gm.graph.nodes
    assert user.args == (cast,)


def test_keep_aten_to_when_copy_true():
    """
    Feature: No-op cast elimination pass
    Description: Preserve aten.to explicit copy=True semantics
    Expectation: aten.to node remains even if dtype does not change
    """
    gm, cast, user, _ = _build_graph_with_cast(
        lambda graph, x: graph.call_function(
            torch.ops.aten.to.device,
            (x, torch.device("cpu"), torch.float32, False, True, None),
        ),
        input_dtype=torch.float32,
        output_dtype=torch.float32,
    )

    eliminate_noop_casts_(gm)

    assert cast in gm.graph.nodes
    assert user.args == (cast,)


def test_keep_to_when_device_changes():
    """
    Feature: No-op cast elimination pass
    Description: Preserve to when metadata indicates a device transfer
    Expectation: to node remains
    """
    gm, cast, user, _ = _build_graph_with_cast(
        lambda graph, x: graph.call_method("to", (x, torch.device("meta"))),
        input_dtype=torch.float32,
        output_dtype=torch.float32,
    )
    cast.meta["example_value"] = _tensor_meta(dtype=torch.float32, device="meta")

    eliminate_noop_casts_(gm)

    assert cast in gm.graph.nodes
    assert user.args == (cast,)


def test_keep_to_when_memory_format_changes():
    """
    Feature: No-op cast elimination pass
    Description: Preserve to when memory_format requests a layout conversion
    Expectation: to node remains
    """
    gm, cast, user, _ = _build_graph_with_cast(
        lambda graph, x: graph.call_method(
            "to", (x,), {"dtype": torch.float32, "memory_format": torch.channels_last}
        ),
        input_dtype=torch.float32,
        output_dtype=torch.float32,
    )

    eliminate_noop_casts_(gm)

    assert cast in gm.graph.nodes
    assert user.args == (cast,)
