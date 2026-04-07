/**
 * Copyright 2026 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <iostream>
#include <cassert>

#include "ir/graph.h"
#include "ir/tensor/tensor.h"
#include "ir/value/value.h"
#include "ir/symbolic/symbolic.h"
#include "ops/op_def/ops_name.h"

using namespace mrt::ir;   // NOLINT(build/namespaces)
using namespace mrt::ops;  // NOLINT(build/namespaces)

void test_deep_copy() {
  std::cout << "Testing Graph DeepCopy functionality..." << std::endl;

  // Create a new graph
  auto graph = MakeIntrusive<Graph>();

  // Create some test nodes
  auto node1 = MakeIntrusive<Node>();
  node1->op = Op_add;
  node1->output = MakeIntrusive<Value>(42.0);  // Double value

  auto node2 = MakeIntrusive<Node>();
  node2->op = Op_mul;
  node2->output = MakeIntrusive<Value>(24.0);  // Double value

  // Add nodes to graph
  graph->nodes.push_back(node1);
  graph->nodes.push_back(node2);

  // Test the DeepCopy functionality
  auto copied_graph = graph->DeepCopy();

  // Verify that the copy was successful
  assert(copied_graph != nullptr);
  assert(copied_graph->nodes.size() == graph->nodes.size());

  std::cout << "Graph DeepCopy test passed!" << std::endl;

  // Test with tensor values
  auto tensor_node = MakeIntrusive<Node>();
  tensor_node->op = Op_empty;
  std::vector<int64_t> shape = {2, 3};
  auto tensor = MakeIntrusive<Tensor>(shape, mrt::DataType::Float32, mrt::hardware::Device());
  tensor_node->output = MakeIntrusive<Value>(tensor);

  auto tensor_graph = MakeIntrusive<Graph>();
  tensor_graph->nodes.push_back(tensor_node);

  auto copied_tensor_graph = tensor_graph->DeepCopy();
  assert(copied_tensor_graph != nullptr);
  assert(copied_tensor_graph->nodes.size() == tensor_graph->nodes.size());

  std::cout << "Tensor DeepCopy test passed!" << std::endl;

  // Test with symbolic expressions
  auto sym_node = MakeIntrusive<Node>();
  sym_node->op = Op_empty;
  auto sym_expr = MakeIntrusive<SymbolicAdd>(MakeIntrusive<SymbolicConst>(5), MakeIntrusive<SymbolicConst>(10));
  sym_node->output = MakeIntrusive<Value>(sym_expr);

  auto sym_graph = MakeIntrusive<Graph>();
  sym_graph->nodes.push_back(sym_node);

  auto copied_sym_graph = sym_graph->DeepCopy();
  assert(copied_sym_graph != nullptr);
  assert(copied_sym_graph->nodes.size() == sym_graph->nodes.size());

  std::cout << "Symbolic Expression DeepCopy test passed!" << std::endl;

  // Test with tuples
  auto tuple_node = MakeIntrusive<Node>();
  tuple_node->op = Op_make_tuple;
  auto val1 = MakeIntrusive<Value>(100);
  auto val2 = MakeIntrusive<Value>(200.0);
  auto tuple_val = MakeIntrusive<Value>(MakeIntrusive<Tuple>(std::vector<ValuePtr>{val1, val2}));
  tuple_node->output = tuple_val;

  auto tuple_graph = MakeIntrusive<Graph>();
  tuple_graph->nodes.push_back(tuple_node);

  auto copied_tuple_graph = tuple_graph->DeepCopy();
  assert(copied_tuple_graph != nullptr);
  assert(copied_tuple_graph->nodes.size() == tuple_graph->nodes.size());

  std::cout << "Tuple DeepCopy test passed!" << std::endl;

  std::cout << "All tests passed successfully!" << std::endl;
}

int main() {
  test_deep_copy();
  return 0;
}
