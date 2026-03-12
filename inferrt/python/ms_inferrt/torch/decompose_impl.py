"""Utilities for per-op FX graph decomposition using FakeTensor + make_fx."""

import operator
from typing import Any, Callable, Dict, List, Optional

import torch
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.fx._lazy_graph_module import _make_graph_module
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.graph_module import GraphModule
from torch.fx.node import Node


def _print_graph(graph: GraphModule, title: str = "Graph") -> None:
    """Print graph nodes in a formatted way for debugging."""
    # Access to _pretty_print_target is intentional for debug dumps.
    print(f"======================{title}======================")
    for node in graph.nodes:
        if node.op == "placeholder":
            print(f"{node.name} = {node.op}")
        elif node.op == "output":
            print(f"output = {node.op}({node.args})")
        elif node.op == "call_function":
            print(
                f"{node.name} = {node.op}"
                f"[{node._pretty_print_target(node.target)}]({node.args})"  # pylint: disable=protected-access
            )
        elif node.op == "call_method":
            print(
                f"{node.name} = {node.op}"
                f"[{node._pretty_print_target(node.target)}]"  # pylint: disable=protected-access
                f"({node.args})({node.kwargs})"
            )
        else:
            print(f"{node.name} = {node.op}")

        example_value = node.meta.get("example_value", None)
        print(f"  example_value: {example_value}")

        val = node.meta.get("val", None)
        print(f"  val: {val}")
    print()


def _should_decompose_getitem(node: Node) -> bool:
    """
    Predicate for deciding whether to decompose an operator.getitem node.

    Skip decomposition when:
      - The 0-th input (base) is a Python tuple/list, or
      - The 0-th input is an FX Node whose example_value is tuple/list, or
      - The 1-st input (index) is a tuple/list that contains all of None,
        Ellipsis, and slice (decomposition would use unsqueeze etc., which does
        not support non-contiguous inputs in that case).
    Otherwise return True to allow decomposition.
    """
    if not node.args:
        # No base; skip decomposition to be conservative.
        return False

    base = node.args[0]

    # Case 1: base is a Python container
    if isinstance(base, (tuple, list)):
        return False

    # Case 2: base is an FX Node whose example_value is a container
    if isinstance(base, Node):
        example_value = base.meta.get("example_value", None)
        if isinstance(example_value, (tuple, list)):
            return False

    # Case 3: index is tuple/list that contains all of None, Ellipsis, and slice
    # (unsqueeze in the decomposed graph does not support non-contiguous inputs)
    if len(node.args) >= 2:
        index = node.args[1]
        if isinstance(index, (tuple, list)):
            has_none = any(item is None for item in index)
            has_ellipsis = any(item is Ellipsis for item in index)
            has_slice = any(isinstance(item, slice) for item in index)
            if has_none and has_ellipsis and has_slice:
                return False

    return True


def _should_decompose_setitem(node: Node) -> bool:
    """
    Predicate for deciding whether to decompose an operator.setitem node.

    Skip make_fx decomposition when the node can be handled directly by
    setitem_process:
      - The 0-th input is an FX Node representing a tensor (type == torch.Tensor
        or example_value is FakeTensor), and
      - The 1-st input (indices) is a slice or (list/tuple), i.e. the two index
        types supported by setitem_impl (tensor_setitem_slice_tensor /
        tensor_setitem_tuple_tensor).

    Return False (do not decompose) when the above holds; otherwise return True
    to run the decomposition flow.
    """
    # Insufficient args; let the decomposition flow handle it.
    if len(node.args) < 2:
        return True

    base = node.args[0]
    index = node.args[1]

    # Only special-case when base is an FX Node.
    if not isinstance(base, Node):
        return True

    example_val = base.meta.get("example_value", None)
    is_tensor_like = (
        getattr(base, "type", None) is torch.Tensor
        or isinstance(example_val, FakeTensor)
    )
    if not is_tensor_like:
        return True

    # setitem_impl currently supports only slice and (list, tuple) indices.
    if isinstance(index, slice):
        return False
    if isinstance(index, (list, tuple)):
        return False

    # Other index types (tensor index, bool mask, etc.) still go through decomposition.
    return True


_DECOMPOSE_FAKE_MODE_OPS: Dict[Any, Optional[Callable[[Node], bool]]] = {
    # key:   op (e.g., operator.setitem, torch.add, "view", ...)
    # value: predicate taking FX Node, returns True to decompose, False to skip.
    #        If value is None, always decompose when key matches.
    operator.setitem: _should_decompose_setitem,
    operator.getitem: _should_decompose_getitem,
}


def _collect_decompose_nodes(
    gm: GraphModule, targets: Dict[Any, Optional[Callable[[Node], bool]]]
) -> List[Node]:
    """
    Collect all call_function nodes whose target is in the given map and whose
    predicate (if provided) returns True.

    - key: op object or name, as used in node.target
    - value: Optional[Callable[[Node], bool]]
        * None -> always decompose when key matches
        * func -> only decompose if func(node) is True
    """
    selected: List[Node] = []
    for node in gm.graph.nodes:
        if node.op != "call_function":
            continue
        if node.target not in targets:
            continue

        predicate = targets[node.target]
        if predicate is None:
            selected.append(node)
        else:
            try:
                if predicate(node):
                    selected.append(node)
            except Exception as exc:  # pylint: disable=broad-except
                # Be conservative: on predicate error, skip this node.
                print(
                    "Skip decompose for node due to predicate exception:",
                    node,
                    "exception:",
                    exc,
                )
    return selected


def _build_single_op_subgraph(node: Node) -> tuple[GraphModule, list[Any]]:
    """
    For a single call_function node, build a tiny subgraph containing only this op,
    with placeholders for tensor arguments and example values attached.
    """
    # Robustness checks for unsupported input patterns
    # kwargs are not supported for per-op decomposition
    if node.kwargs:
        raise RuntimeError(
            "decompose_ops_with_fake_mode does not support kwargs inputs; "
            f"found kwargs={node.kwargs} for node {node.op}[{node.target}]"
        )

    subgraph = torch.fx.Graph()
    arg_nodes: list[Any] = []
    example_args: list[Any] = []

    for i, arg in enumerate(node.args):
        if isinstance(arg, Node):
            example_value = arg.meta.get("example_value", None)
            placeholder = subgraph.placeholder(f"arg_{i}")
            placeholder.meta = {"example_value": example_value}
            arg_nodes.append(placeholder)
            example_args.append(example_value)
        else:
            arg_nodes.append(arg)

    single_op_node = subgraph.call_function(node.target, tuple(arg_nodes), {})
    single_op_node.meta = node.meta.copy()
    subgraph.output(single_op_node)

    subgraph_gm = _make_graph_module({}, subgraph, "single_op_subgraph")
    return subgraph_gm, example_args


def _trace_subgraph_with_fake_tensor(
    subgraph_gm: GraphModule,
    example_args: list[Any],
    fallback_gm: Optional[GraphModule] = None,
) -> Optional[GraphModule]:
    """
    Run make_fx under FakeTensorMode on the given subgraph.

    If tracing fails, fall back to the provided fallback_gm (or the original
    subgraph_gm if no explicit fallback is given).
    """
    fake_mode = FakeTensorMode(allow_non_fake_inputs=True)
    try:
        with fake_mode:
            subgraph_gm_replaced = make_fx(subgraph_gm)(*example_args)
    except Exception as e:  # pylint: disable=broad-except
        print("Failed to make fx for graph:")
        print(subgraph_gm.graph)
        print("exception:", e)
        # On failure, signal caller to skip replacement and keep original node
        return fallback_gm

    return subgraph_gm_replaced


def _build_placeholder_node_map(
    original_node: Node, traced_subgraph_gm: GraphModule
) -> dict[Node, Node]:
    """
    Map placeholder nodes in traced_subgraph_gm back to the original node.args.
    """
    node_map: dict[Node, Node] = {}
    placeholder_nodes = [
        n for n in traced_subgraph_gm.graph.nodes if n.op == "placeholder"
    ]

    placeholder_idx = 0
    for arg in original_node.args:
        if isinstance(arg, Node):
            if placeholder_idx < len(placeholder_nodes):
                node_map[placeholder_nodes[placeholder_idx]] = arg
                placeholder_idx += 1
    return node_map


def _map_args_with_node_map(arg: Any, node_map: dict[Node, Node]) -> Any:
    """
    Recursively map FX Node arguments/structures using node_map.
    """
    if isinstance(arg, Node):
        return node_map.get(arg, arg)
    if isinstance(arg, (list, tuple)):
        return type(arg)(_map_args_with_node_map(a, node_map) for a in arg)
    if isinstance(arg, dict):
        return {k: _map_args_with_node_map(v, node_map) for k, v in arg.items()}
    return arg


def _inline_traced_subgraph(
    gm: GraphModule,
    original_node: Node,
    traced_subgraph_gm: GraphModule,
    node_map: dict[Node, Node],
) -> None:
    """
    Inline the traced subgraph back into the main graph, replacing the original
    node.
    """
    with gm.graph.inserting_before(original_node):
        for sub_node in traced_subgraph_gm.graph.nodes:
            if sub_node.op in ("placeholder", "output"):
                continue

            if sub_node.op == "call_function":
                new_args = _map_args_with_node_map(sub_node.args, node_map)
                new_kwargs = _map_args_with_node_map(sub_node.kwargs, node_map)
                new_node = gm.graph.call_function(sub_node.target, new_args, new_kwargs)
            elif sub_node.op == "call_method":
                new_args = _map_args_with_node_map(sub_node.args, node_map)
                new_kwargs = _map_args_with_node_map(sub_node.kwargs, node_map)
                new_node = gm.graph.call_method(sub_node.target, new_args, new_kwargs)
            else:
                continue

            new_node.meta = {"example_value": sub_node.meta.get("val", None)}
            node_map[sub_node] = new_node

    # Redirect users of the original node to the inlined subgraph output.
    # We only support the common case where the traced subgraph has a single
    # tensor output (i.e., output_node.args[0] is a Node). For more complex
    # multi-output patterns, fall back to keeping the original node to avoid
    # inconsistent use lists / assertions inside FX.
    output_node = traced_subgraph_gm.graph.output_node()
    out_args = output_node.args
    if not isinstance(out_args, tuple) or len(out_args) != 1:
        print(
            "Skip inlining for node due to unsupported output structure: "
            f"{output_node.op}, args={output_node.args}"
        )
        return

    out_val = out_args[0]

    if isinstance(out_val, list):
        print("output value is list, element 0:", out_val[0])

    if not isinstance(out_val, Node) or out_val not in node_map:
        print(
            "Skip inlining for node due to missing mapping for output value: "
            f"{out_val} node map[{node_map}]"
        )
        return

    replacement = node_map[out_val]
    original_node.replace_all_uses_with(replacement)
    gm.graph.erase_node(original_node)


def _decompose_ops_with_fake_mode(gm: GraphModule) -> None:
    """
    Decompose selected operators in the graph by:
      1) Extracting them into tiny subgraphs,
      2) Tracing with make_fx under FakeTensorMode to get fine-grained ops,
      3) Inlining the traced subgraphs back into the original graph.
    """
    target_nodes = _collect_decompose_nodes(gm, _DECOMPOSE_FAKE_MODE_OPS)
    if not target_nodes:
        return

    for node in target_nodes:
        subgraph_gm, example_args = _build_single_op_subgraph(node)
        traced_subgraph_gm = _trace_subgraph_with_fake_tensor(
            subgraph_gm, example_args, fallback_gm=None
        )
        # If tracing failed, skip this node and keep the original op
        if traced_subgraph_gm is None:
            print("Failed to decompose for node:", node)
            continue
        node_map = _build_placeholder_node_map(node, traced_subgraph_gm)
        _inline_traced_subgraph(gm, node, traced_subgraph_gm, node_map)
