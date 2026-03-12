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

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/optional.h>
#include <nanobind/make_iterator.h>
#include <sstream>
#include <vector>
#include <string>
#include <optional>

#include "common/intrusive_ptr_caster.h"
#include "runtime/executor/executor.h"
#include "ops/utils/op_support.h"
#include "ir/graph.h"
#include "ir/value/value.h"
#include "ir/tensor/tensor.h"
#include "ir/common/dtype.h"
#include "hardware/device.h"
#include "ops/op_def/ops_name.h"
#include "ir/symbolic/symbolic.h"

namespace nb = nanobind;
namespace ir = mrt::ir;
namespace hardware = mrt::hardware;
using mrt::runtime::GraphExecutor;

NB_MODULE(_ms_inferrt_ir, m) {  // #lizard forgives
  m.doc() = "Python binding for InferRT";

  nb::enum_<mrt::ops::Op>(m, "Op")
#define OP(O) .value(#O, mrt::ops::Op_##O)
#include "ops/op_def/ops.list"
#undef OP
    .export_values();

  nb::class_<ir::DataType> DataType(m, "DataType");
  DataType.def(nb::init<ir::DataType::Type>())
    .def_rw("value", &ir::DataType::value)
    .def_static("from_string", ir::DataType::FromString, nb::arg("dtype_str"),
                "Convert a dtype string to a DataType object.")
    .def_static(
      "convert_str_to_int",
      [](const std::string &dtype_str) { return static_cast<int>(ir::DataType::FromString(dtype_str).value); },
      nb::arg("dtype_str"), "Convert a dtype string to an integer of enum DataType.Type");

  nb::enum_<ir::DataType::Type>(DataType, "Type")
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
    .value("QInt8", ir::DataType::Type::QInt8)
    .value("QUInt4x2", ir::DataType::Type::QUInt4x2)
    .value("Bool", ir::DataType::Type::Bool)
    .export_values();

  nb::enum_<hardware::DeviceType>(m, "DeviceType")
    .value("CPU", hardware::DeviceType::CPU)
    .value("NPU", hardware::DeviceType::NPU)
    .export_values();

  nb::class_<hardware::Device>(m, "Device")
    .def(nb::init<hardware::DeviceType, hardware::DeviceIndex>(), nb::arg("type") = hardware::DeviceType::CPU,
         nb::arg("index") = -1)
    .def_rw("type", &hardware::Device::type)
    .def_rw("index", &hardware::Device::index);

  nb::class_<ir::SymbolicExpr>(m, "SymbolicExpr", nb::dynamic_attr())
    .def("__repr__", &ir::SymbolicExpr::ToString)
    .def("__add__", [](const ir::SymbolicExprPtr &a, const ir::SymbolicExprPtr &b) { return a + b; })
    .def("__mul__", [](const ir::SymbolicExprPtr &a, const ir::SymbolicExprPtr &b) { return a * b; })
    .def("__truediv__", [](const ir::SymbolicExprPtr &a, const ir::SymbolicExprPtr &b) { return a / b; })
    .def("__mod__", [](const ir::SymbolicExprPtr &a, const ir::SymbolicExprPtr &b) { return a % b; })
    .def("__floordiv__", [](const ir::SymbolicExprPtr &a, const ir::SymbolicExprPtr &b) { return ir::FloorDiv(a, b); })
    .def("__ceildiv__", [](const ir::SymbolicExprPtr &a, const ir::SymbolicExprPtr &b) { return ir::CeilDiv(a, b); });

  nb::class_<ir::SymbolicVar, ir::SymbolicExpr>(m, "SymbolicVar", nb::dynamic_attr())
    .def(nb::new_([](const std::string &name) { return ir::MakeIntrusive<ir::SymbolicVar>(name); }), nb::arg("name"))
    .def("set_value", &ir::SymbolicVar::SetValue, nb::arg("value"));

  nb::class_<ir::SymbolicConst, ir::SymbolicExpr>(m, "SymbolicConst", nb::dynamic_attr())
    .def(nb::new_([](int64_t value) { return ir::MakeIntrusive<ir::SymbolicConst>(value); }), nb::arg("value"));

  nb::class_<ir::Tensor>(m, "Tensor", nb::dynamic_attr())
    .def(nb::new_(
           [](const std::vector<int64_t> &shape, const ir::DataType &dtype, std::optional<hardware::Device> device) {
             return ir::MakeIntrusive<ir::Tensor>(shape, dtype,
                                                  device.value_or(hardware::Device(hardware::DeviceType::CPU, -1)));
           }),
         nb::arg("shape"), nb::arg("dtype"), nb::arg("device") = nb::none())
    .def_prop_rw(
      "shape", [](const ir::Tensor &t) { return t.Shape(); },
      [](ir::Tensor &t, const std::vector<int64_t> &s) { t.SetShape(s); })
    .def_prop_rw("symbolic_shape", &ir::Tensor::GetSymbolicShape, &ir::Tensor::SetSymbolicShape)
    .def_prop_rw("dtype", &ir::Tensor::Dtype, &ir::Tensor::SetDtype)
    .def("__repr__", [](const ir::Tensor &t) {
      std::stringstream ss;
      ss << t;
      return ss.str();
    });

  nb::class_<ir::Tuple>(m, "Tuple", nb::dynamic_attr())
    .def(nb::new_([](std::vector<ir::ValuePtr> elements) { return ir::MakeIntrusive<ir::Tuple>(std::move(elements)); }))
    .def("__len__", &ir::Tuple::Size)
    .def("__getitem__", &ir::Tuple::operator[], nb::rv_policy::reference)
    .def("__setitem__", [](ir::Tuple &t, size_t i, const ir::Value &v) { *(t[i]) = v; })
    .def(
      "__iter__",
      [](const ir::Tuple &t) { return nb::make_iterator(nb::type<ir::Tuple>(), "iterator", t.begin(), t.end()); },
      nb::keep_alive<0, 1>())
    .def("__repr__", [](const ir::TuplePtr &t) {
      std::stringstream ss;
      ss << t;
      return ss.str();
    });

  nb::class_<ir::Value>(m, "Value", nb::dynamic_attr())
    .def(nb::new_([]() { return ir::MakeIntrusive<ir::Value>(); }))
    .def(nb::new_([](const ir::TensorPtr &v) { return ir::MakeIntrusive<ir::Value>(v); }))
    .def(nb::new_([](double v) { return ir::MakeIntrusive<ir::Value>(v); }))
    .def(nb::new_([](bool v) { return ir::MakeIntrusive<ir::Value>(v); }))
    .def(nb::new_([](int64_t v) { return ir::MakeIntrusive<ir::Value>(v); }))
    .def(nb::new_([](std::string v) { return ir::MakeIntrusive<ir::Value>(std::move(v)); }))
    .def(nb::new_([](const ir::TuplePtr &v) { return ir::MakeIntrusive<ir::Value>(v); }))
    .def(nb::new_([](const ir::SymbolicExprPtr &v) { return ir::MakeIntrusive<ir::Value>(v); }))
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

  nb::class_<ir::Node>(m, "Node", nb::dynamic_attr())
    .def_prop_rw(
      "output", [](const ir::Node &node) { return node.output; },
      [](ir::Node &node, const ir::Value &value) { *(node.output) = value; });

  nb::class_<GraphExecutor>(m, "GraphExecutor", nb::dynamic_attr())
    .def(nb::init<>())
    .def("begin_graph", &GraphExecutor::BeginGraph, nb::arg("name"))
    .def("end_graph", &GraphExecutor::EndGraph)
    .def("opt_graph", &GraphExecutor::OptGraph)
    .def("build_kernels", &GraphExecutor::BuildKernels)
    .def("build_executor", &GraphExecutor::BuildExecutor)
    .def("run_graph", &GraphExecutor::RunGraph, nb::arg("is_dynamic") = false)
    .def("dump_graph", &GraphExecutor::DumpGraph)
    .def("record_tensor_ref_count", &GraphExecutor::RecordTensorRefCount)
    .def("add_return_node", &GraphExecutor::AddReturnNode)
    .def("add_parameter_node", &GraphExecutor::AddParameterNode, nb::arg("value") = nullptr, nb::rv_policy::reference)
    .def("add_input_node", &GraphExecutor::AddInputNode, nb::arg("value") = nullptr, nb::rv_policy::reference)
    .def("add_op_node", &GraphExecutor::AddOpNode, nb::arg("op"), nb::arg("inputs"), nb::arg("output") = nullptr,
         nb::rv_policy::reference)
    .def("add_value_node", &GraphExecutor::AddValueNode, nb::arg("value") = nullptr, nb::rv_policy::reference);

  m.def(
    "check_op_support",
    [](const std::string &opName, const ir::ValuePtr &outputValue, const std::vector<ir::ValuePtr> &inputValues) {
      const auto result = mrt::runtime::CheckOpSupport(opName, outputValue, inputValues);
      return nb::make_tuple(static_cast<int32_t>(result.status), result.message);
    },
    nb::arg("op_name"), nb::arg("output_value"), nb::arg("input_values"));
}
