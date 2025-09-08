/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sstream>

#include "runtime/executor/executor.h"
#include "ir/graph.h"
#include "ir/value/value.h"
#include "ops/op_def/ops_name.h"

namespace py = pybind11;
using namespace mrt;
using namespace mrt::runtime;

PYBIND11_DECLARE_HOLDER_TYPE(T, ir::IntrusivePtr<T>, true);

PYBIND11_MODULE(_mrt_ir, m) {
  m.doc() = "Python binding for DA IR";

  py::enum_<ops::Op>(m, "Op")
#define OP(O) .value(#O, ops::Op_##O)
#include "ops/op_def/ops.list"
#undef OP
    .export_values();

  py::class_<ir::Tensor, ir::TensorPtr>(m, "Tensor")
    .def_property_readonly("shape", &ir::Tensor::Shape)
    .def_property_readonly("dtype", &ir::Tensor::Dtype)
    .def("__repr__", [](const ir::Tensor &t) {
      std::stringstream ss;
      ss << t;
      return ss.str();
    });

  py::class_<ir::Tuple, ir::TuplePtr>(m, "Tuple")
    .def(py::init<std::vector<ir::ValuePtr>>())
    .def("__len__", &ir::Tuple::Size)
    .def("__getitem__", &ir::Tuple::operator[], py::return_value_policy::reference)
    .def("__iter__", [](const ir::Tuple &t) { return py::make_iterator(t.begin(), t.end()); }, py::keep_alive<0, 1>())
    .def("__repr__", [](const ir::TuplePtr &t) {
      std::stringstream ss;
      ss << t;
      return ss.str();
    });

  py::class_<ir::Value, ir::ValuePtr>(m, "Value")
    .def(py::init<>())
    .def(py::init<ir::TensorPtr>())
    .def(py::init<double>())
    .def(py::init<int64_t>())
    .def(py::init<bool>())
    .def(py::init<std::string>())
    .def(py::init<ir::TuplePtr>())
    .def("is_tensor", &ir::Value::IsTensor)
    .def("is_tuple", &ir::Value::IsTuple)
    .def("is_double", &ir::Value::IsDouble)
    .def("is_int", &ir::Value::IsInt)
    .def("is_bool", &ir::Value::IsBool)
    .def("is_string", &ir::Value::IsString)
    .def("is_none", &ir::Value::IsNone)
    .def("to_tensor", &ir::Value::ToTensor)
    .def("to_tuple", &ir::Value::ToTuple)
    .def("to_double", &ir::Value::ToDouble)
    .def("to_int", &ir::Value::ToInt)
    .def("to_bool", &ir::Value::ToBool)
    .def("to_string", &ir::Value::ToString)
    .def("__repr__", [](const ir::Value &v) {
      std::stringstream ss;
      ss << v;
      return ss.str();
    });

  py::class_<ir::Node, ir::NodePtr>(m, "Node").def_property_readonly(
    "output", [](const ir::NodePtr &node) { return node->output; });

  py::class_<GraphExecutor>(m, "GraphExecutor")
    .def(py::init<>())
    .def("begin_graph", &GraphExecutor::BeginGraph, py::arg("name"))
    .def("end_graph", &GraphExecutor::EndGraph)
    .def("opt_graph", &GraphExecutor::OptGraph)
    .def("build_kernels", &GraphExecutor::BuildKernels)
    .def("run_graph", &GraphExecutor::RunGraph, py::arg("is_dynamic") = false)
    .def("dump_graph", &GraphExecutor::DumpGraph)
    .def("record_tensor_ref_count", &GraphExecutor::RecordTensorRefCount)
    .def("add_return", &GraphExecutor::AddReturn, py::return_value_policy::reference)
    .def("add_parameter", &GraphExecutor::AddParameter, py::arg("param"))
    .def("add_op_node", &GraphExecutor::AddOpNode, py::arg("op"), py::arg("inputs"), py::return_value_policy::reference)
    .def("add_value_node", &GraphExecutor::AddValueNode, py::arg("value"), py::return_value_policy::reference);
}
