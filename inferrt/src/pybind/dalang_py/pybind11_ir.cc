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
#include <pybind11/numpy.h>
#include <sstream>

#include "runtime/executor/executor.h"
#include "ir/graph.h"
#include "ops/op_def/ops_name.h"

namespace py = pybind11;
using namespace mrt;
using namespace mrt::runtime;

static ir::ValuePtr ConvertBufferInfoToValue(const py::buffer_info &info) {
  ir::DataType type;
  if (info.format == py::format_descriptor<float>::format())
    type = ir::DataType::Float32;
  else if (info.format == py::format_descriptor<double>::format())
    type = ir::DataType::Float64;
  else if (info.format == py::format_descriptor<int32_t>::format())
    type = ir::DataType::Int32;
  else if (info.format == py::format_descriptor<int64_t>::format())
    type = ir::DataType::Int64;
  else if (info.format == py::format_descriptor<int16_t>::format())
    type = ir::DataType::Int16;
  else if (info.format == py::format_descriptor<bool>::format())
    type = ir::DataType::Bool;
  else
    throw std::runtime_error("Unsupported numpy dtype for buffer conversion");
  std::vector<int64_t> dims(info.shape.begin(), info.shape.end());
  auto tensor = ir::Tensor(info.ptr, dims, type, hardware::Device(hardware::DeviceType::CPU, 0));
  return ir::MakeIntrusive<ir::Value>(std::move(tensor));
}

static py::array ConvertValueToNumpyArray(ir::ValuePtr value) {
  auto tensor = value->ToTensor();
  std::string format;
  switch (tensor->Dtype()) {
    case ir::DataType::Float32:
      format = py::format_descriptor<float>::format();
      break;
    case ir::DataType::Int32:
      format = py::format_descriptor<int32_t>::format();
      break;
    case ir::DataType::Int64:
      format = py::format_descriptor<int64_t>::format();
      break;
    case ir::DataType::Float64:
      format = py::format_descriptor<double>::format();
      break;
    case ir::DataType::Bool:
      format = py::format_descriptor<bool>::format();
      break;
    case ir::DataType::Int16:
      format = py::format_descriptor<int16_t>::format();
      break;
    default:
      throw std::runtime_error("Unsupported dtype for numpy conversion");
  }
  std::vector<size_t> shape(tensor->Shape().begin(), tensor->Shape().end());
  auto result = py::array(py::dtype(format), shape, tensor->DataPtr());
  return result;
}

static std::vector<py::array> ConvertValueToNumpyArrayList(ir::ValuePtr value) {
  std::vector<py::array> result;
  for (auto &item : *(value->ToTuple())) {
    (void)result.emplace_back(ConvertValueToNumpyArray(item));
  }
  return result;
}

PYBIND11_MODULE(_dairpy, m) {
  m.doc() = "Python binding for DA IR";

  py::enum_<ops::Op>(m, "Op")
#define OP(O) .value(#O, ops::Op_##O)
#include "ops/op_def/ops.list"
#undef OP
    .export_values();

  py::class_<ir::Node, ir::NodePtr>(m, "Node")
    .def("update_data",
         [](ir::Node &self, py::buffer b) {
           py::buffer_info info = b.request();
           self.output = ConvertBufferInfoToValue(b.request());
         })
    .def("numpy", [](ir::Node &self) -> py::array { return ConvertValueToNumpyArray(self.output); })
    .def("list", [](ir::Node &self) -> std::vector<py::array> { return ConvertValueToNumpyArrayList(self.output); });

  py::class_<GraphExecutor>(m, "GraphExecutor")
    .def(py::init<>())
    .def("begin_graph", &GraphExecutor::BeginGraph, py::arg("name"))
    .def("end_graph", &GraphExecutor::EndGraph)
    .def("opt_graph", &GraphExecutor::OptGraph)
    .def("build_kernels", &GraphExecutor::BuildKernels)
    .def("run_graph", &GraphExecutor::RunGraph, py::arg("is_dynamic") = false)
    .def("dump_graph", &GraphExecutor::DumpGraph)
    .def("free_graph_outputs", &GraphExecutor::FreeGraphOutputs)
    .def("record_tensor_ref_count", &GraphExecutor::RecordTensorRefCount)
    .def("add_return", &GraphExecutor::AddReturn, py::return_value_policy::reference)
    .def("add_parameter", &GraphExecutor::AddParameter, py::arg("tensor"))
    .def(
      "add_op",
      [](GraphExecutor &self, ops::Op op, const std::vector<ir::NodePtr> &inputs) -> ir::NodePtr {
        return self.AddTensor(op, inputs);
      },
      py::arg("op"), py::arg("inputs"), py::return_value_policy::reference)
    .def(
      "add_const",
      [](GraphExecutor &self, const py::buffer b) -> ir::NodePtr {
        auto tensor = self.AddTensor();
        tensor->output = ConvertBufferInfoToValue(b.request());
        return tensor;
      },
      py::arg("tensor"), py::return_value_policy::reference);
}
