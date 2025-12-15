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
#include <vector>
#include <string>

#include "runtime/executor/executor.h"
#include "ir/graph.h"
#include "ir/value/value.h"
#include "ir/tensor/tensor.h"
#include "ir/common/dtype.h"
#include "hardware/device.h"
#include "ops/op_def/ops_name.h"
#include "ir/symbolic/symbolic.h"

namespace py = pybind11;
namespace ir = mrt::ir;
namespace hardware = mrt::hardware;
using mrt::runtime::GraphExecutor;

PYBIND11_DECLARE_HOLDER_TYPE(T, ir::IntrusivePtr<T>, true);

PYBIND11_MODULE(_mrt_ir, m) {
  m.doc() = "Python binding for MRT";

  py::enum_<mrt::ops::Op>(m, "Op")
#define OP(O) .value(#O, mrt::ops::Op_##O)
#include "ops/op_def/ops.list"
#undef OP
    .export_values();

  py::class_<ir::DataType> DataType(m, "DataType");
  DataType.def(py::init<ir::DataType::Type>())
    .def_readwrite("value", &ir::DataType::value)
    .def_static("from_string", ir::DataType::FromString, py::arg("dtype_str"),
                "Convert a dtype string to a DataType object.")
    .def_static(
      "convert_str_to_int",
      [](const std::string &dtype_str) { return static_cast<int>(ir::DataType::FromString(dtype_str).value); },
      py::arg("dtype_str"), "Convert a dtype string to an integer of enum DataType.Type");

  py::enum_<ir::DataType::Type>(DataType, "Type")
    .value("Unknown", ir::DataType::Type::Unknown)
    .value("Float16", ir::DataType::Type::Float16)
    .value("BFloat16", ir::DataType::Type::BFloat16)
    .value("Float32", ir::DataType::Type::Float32)
    .value("Float64", ir::DataType::Type::Float64)
    .value("Complex64", ir::DataType::Type::Complex64)
    .value("Int8", ir::DataType::Type::Int8)
    .value("Int16", ir::DataType::Type::Int16)
    .value("Int32", ir::DataType::Type::Int32)
    .value("Int64", ir::DataType::Type::Int64)
    .value("UInt8", ir::DataType::Type::UInt8)
    .value("Bool", ir::DataType::Type::Bool)
    .export_values();

  py::enum_<hardware::DeviceType>(m, "DeviceType")
    .value("CPU", hardware::DeviceType::CPU)
    .value("NPU", hardware::DeviceType::NPU)
    .export_values();

  py::class_<hardware::Device>(m, "Device")
    .def(py::init<hardware::DeviceType, hardware::DeviceIndex>(), py::arg("type") = hardware::DeviceType::CPU,
         py::arg("index") = -1)
    .def_readwrite("type", &hardware::Device::type)
    .def_readwrite("index", &hardware::Device::index);

  py::class_<ir::SymbolicExpr, ir::SymbolicExprPtr>(m, "SymbolicExpr")
    .def("__repr__", &ir::SymbolicExpr::ToString)
    .def("__add__", [](const ir::SymbolicExprPtr &a, const ir::SymbolicExprPtr &b) { return a + b; })
    .def("__mul__", [](const ir::SymbolicExprPtr &a, const ir::SymbolicExprPtr &b) { return a * b; })
    .def("__truediv__", [](const ir::SymbolicExprPtr &a, const ir::SymbolicExprPtr &b) { return a / b; })
    .def("__mod__", [](const ir::SymbolicExprPtr &a, const ir::SymbolicExprPtr &b) { return a % b; })
    .def("__floordiv__", [](const ir::SymbolicExprPtr &a, const ir::SymbolicExprPtr &b) { return ir::FloorDiv(a, b); })
    .def("__ceildiv__", [](const ir::SymbolicExprPtr &a, const ir::SymbolicExprPtr &b) { return ir::CeilDiv(a, b); });

  py::class_<ir::SymbolicVar, ir::SymbolicExpr, ir::IntrusivePtr<ir::SymbolicVar>>(m, "SymbolicVar")
    .def(py::init<const std::string &>(), py::arg("name"))
    .def("set_value", &ir::SymbolicVar::SetValue, py::arg("value"));

  py::class_<ir::SymbolicConst, ir::SymbolicExpr, ir::IntrusivePtr<ir::SymbolicConst>>(m, "SymbolicConst")
    .def(py::init<int64_t>(), py::arg("value"));

  py::class_<ir::Tensor, ir::TensorPtr>(m, "Tensor")
    .def(py::init([](const std::vector<int64_t> &shape, const ir::DataType &dtype, const hardware::Device &device) {
           return ir::MakeIntrusive<ir::Tensor>(shape, dtype, device);
         }),
         py::arg("shape"), py::arg("dtype"), py::arg("device") = hardware::Device(hardware::DeviceType::CPU, -1))
    .def_property("shape", py::overload_cast<>(&ir::Tensor::Shape, py::const_),
                  py::overload_cast<const std::vector<int64_t> &>(&ir::Tensor::SetShape))
    .def_property("symbolic_shape", &ir::Tensor::GetSymbolicShape, &ir::Tensor::SetSymbolicShape)
    .def_property("dtype", &ir::Tensor::Dtype, &ir::Tensor::SetDtype)
    .def("__repr__", [](const ir::Tensor &t) {
      std::stringstream ss;
      ss << t;
      return ss.str();
    });

  py::class_<ir::Tuple, ir::TuplePtr>(m, "Tuple")
    .def(py::init<std::vector<ir::ValuePtr>>())
    .def("__len__", &ir::Tuple::Size)
    .def("__getitem__", &ir::Tuple::operator[], py::return_value_policy::reference)
    .def("__setitem__", [](ir::Tuple &t, size_t i, const ir::Value &v) { *(t[i]) = v; })
    .def(
      "__iter__", [](const ir::Tuple &t) { return py::make_iterator(t.begin(), t.end()); }, py::keep_alive<0, 1>())
    .def("__repr__", [](const ir::TuplePtr &t) {
      std::stringstream ss;
      ss << t;
      return ss.str();
    });

  py::class_<ir::Value, ir::ValuePtr>(m, "Value")
    .def(py::init<>())
    .def(py::init<const ir::TensorPtr &>())
    .def(py::init<double>())
    .def(py::init<bool>())
    .def(py::init<int64_t>())
    .def(py::init<std::string>())
    .def(py::init<const ir::TuplePtr &>())
    .def(py::init<const ir::SymbolicExprPtr &>())
    .def("is_tensor", &ir::Value::IsTensor)
    .def("is_tuple", &ir::Value::IsTuple)
    .def("is_symbol", &ir::Value::IsSymbol)
    .def("is_double", &ir::Value::IsDouble)
    .def("is_int", &ir::Value::IsInt)
    .def("is_bool", &ir::Value::IsBool)
    .def("is_string", &ir::Value::IsString)
    .def("is_none", &ir::Value::IsNone)
    .def("to_tensor", &ir::Value::ToTensor)
    .def("to_tuple", &ir::Value::ToTuple)
    .def("to_symbol", &ir::Value::ToSymbol)
    .def("to_double", &ir::Value::ToDouble)
    .def("to_int", &ir::Value::ToInt)
    .def("to_bool", &ir::Value::ToBool)
    .def("to_string", &ir::Value::ToString)
    .def("__repr__", [](const ir::Value &v) {
      std::stringstream ss;
      ss << v;
      return ss.str();
    });

  py::class_<ir::Node, ir::NodePtr>(m, "Node").def_property(
    "output", [](const ir::NodePtr &node) { return node->output; },
    [](ir::NodePtr &node, const ir::Value &value) { *(node->output) = value; });

  py::class_<GraphExecutor>(m, "GraphExecutor")
    .def(py::init<>())
    .def("begin_graph", &GraphExecutor::BeginGraph, py::arg("name"))
    .def("end_graph", &GraphExecutor::EndGraph)
    .def("opt_graph", &GraphExecutor::OptGraph)
    .def("build_kernels", &GraphExecutor::BuildKernels)
    .def("build_executor", &GraphExecutor::BuildExecutor)
    .def("run_graph", &GraphExecutor::RunGraph, py::arg("is_dynamic") = false)
    .def("dump_graph", &GraphExecutor::DumpGraph)
    .def("record_tensor_ref_count", &GraphExecutor::RecordTensorRefCount)
    .def("add_return", &GraphExecutor::AddReturn, py::return_value_policy::reference)
    .def("add_parameter", &GraphExecutor::AddParameter, py::arg("param"))
    .def("add_op_node", &GraphExecutor::AddOpNode, py::arg("op"), py::arg("inputs"), py::arg("output"),
         py::return_value_policy::reference)
    .def("add_value_node", &GraphExecutor::AddValueNode, py::arg("value"), py::return_value_policy::reference);
}
