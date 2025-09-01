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

#include <torch/extension.h>

#include "runtime/executor/executor.h"
#include "ir/graph.h"
#include "ir/value/value.h"
#include "ops/op_def/ops_name.h"

namespace py = pybind11;
using namespace mrt;
using namespace mrt::runtime;

namespace {
// DataType conversion utilities
ir::DataType FromTorchDType(const at::ScalarType &type) {
  switch (type) {
    case at::kFloat:
      return ir::DataType::Float32;
    case at::kDouble:
      return ir::DataType::Float64;
    case at::kInt:
      return ir::DataType::Int32;
    case at::kLong:
      return ir::DataType::Int64;
    case at::kShort:
      return ir::DataType::Int16;
    case at::kBool:
      return ir::DataType::Bool;
    default:
      LOG_EXCEPTION << "Unsupported at::ScalarType for conversion to ir::DataType";
      return ir::DataType::Unknown;
  }
}

at::ScalarType ToTorchDType(ir::DataType type) {
  switch (type) {
    case ir::DataType::Float32:
      return at::kFloat;
    case ir::DataType::Float64:
      return at::kDouble;
    case ir::DataType::Int32:
      return at::kInt;
    case ir::DataType::Int64:
      return at::kLong;
    case ir::DataType::Int16:
      return at::kShort;
    case ir::DataType::Bool:
      return at::kBool;
    default:
      LOG_EXCEPTION << "Unsupported ir::DataType for conversion to at::ScalarType";
      return at::kFloat;
  }
}

// Device conversion utilities
hardware::Device FromTorchDevice(const at::Device &device) {
  hardware::DeviceType deviceType;
  switch (device.type()) {
    case at::DeviceType::CPU:
      deviceType = hardware::DeviceType::CPU;
      break;
    default:
      LOG_EXCEPTION << "Unsupported torch::Device for conversion to hardware::Device";
  }
  return hardware::Device(deviceType, device.index());
}

at::Device ToTorchDevice(const hardware::Device device) {
  at::DeviceType deviceType;
  switch (device.type) {
    case hardware::DeviceType::CPU:
      deviceType = at::kCPU;
      break;
    default:
      LOG_EXCEPTION << "Unsupported hardware::DeviceType for conversion to torch::Device";
  }
  return at::Device(deviceType, device.index);
}

// Forward declaration for recursive conversion
py::object to_python(const ir::ValuePtr &value);
ir::ValuePtr from_python(const py::handle &obj);

// Tensor conversion utilities
ir::ValuePtr from_torch(const at::Tensor &atTensor) {
  ir::DataType type = FromTorchDType(atTensor.scalar_type());
  std::vector<int64_t> shape(atTensor.sizes().begin(), atTensor.sizes().end());
  void *data = atTensor.data_ptr();
  auto device = FromTorchDevice(atTensor.device());
  auto tensor = ir::Tensor(data, shape, type, device);
  return ir::MakeIntrusive<ir::Value>(std::move(tensor));
}

at::Tensor to_torch(const ir::ValuePtr &value) {
  auto tensor = value->ToTensor();
  CHECK_IF_NULL(tensor);
  auto options = at::TensorOptions().dtype(ToTorchDType(tensor->Dtype())).device(ToTorchDevice(tensor->GetDevice()));
  value->AddRef();
  return at::from_blob(
    tensor->DataPtr(), tensor->Shape(), tensor->Strides(), [ptr = value.get()](void *) { ptr->DecRef(); }, options);
}

// New conversion functions
ir::ValuePtr from_python(const py::handle &obj) {
  if (THPVariable_Check(obj.ptr())) {
    return from_torch(obj.cast<at::Tensor>());
  }
  if (py::isinstance<py::tuple>(obj) || py::isinstance<py::list>(obj)) {
    auto py_tuple = obj.cast<py::tuple>();
    std::vector<ir::ValuePtr> elements;
    elements.reserve(py_tuple.size());
    for (const auto &elem : py_tuple) {
      (void)elements.emplace_back(from_python(elem));
    }
    return ir::MakeIntrusive<ir::Value>(ir::Tuple(std::move(elements)));
  }
  if (py::isinstance<py::int_>(obj)) {
    return ir::MakeIntrusive<ir::Value>(obj.cast<int64_t>());
  }
  if (py::isinstance<py::float_>(obj)) {
    return ir::MakeIntrusive<ir::Value>(obj.cast<double>());
  }
  if (py::isinstance<py::bool_>(obj)) {
    return ir::MakeIntrusive<ir::Value>(obj.cast<bool>());
  }
  if (py::isinstance<py::str>(obj)) {
    return ir::MakeIntrusive<ir::Value>(obj.cast<std::string>());
  }
  if (obj.is_none()) {
    return ir::MakeIntrusive<ir::Value>();
  }
  LOG_EXCEPTION << "Unsupported python type for conversion to ir::Value: " << py::str(obj);
  return nullptr;
}

py::object to_python(const ir::ValuePtr &value) {
  if (!value) {
    return py::none();
  }
  if (value->IsTensor()) {
    return py::cast(to_torch(value));
  }
  if (value->IsTuple()) {
    const auto *tuple = value->ToTuple();
    py::tuple py_tuple(tuple->Size());
    for (size_t i = 0; i < tuple->Size(); ++i) {
      py_tuple[i] = to_python((*tuple)[i]);
    }
    return py_tuple;
  }
  if (value->IsInt()) {
    return py::cast(value->ToInt());
  }
  if (value->IsDouble()) {
    return py::cast(value->ToDouble());
  }
  if (value->IsBool()) {
    return py::cast(value->ToBool());
  }
  if (value->IsString()) {
    return py::cast(*value->ToString());
  }
  if (value->IsNone()) {
    return py::none();
  }
  LOG_EXCEPTION << "Unsupported ir::Value for conversion to python object: " << value;
  return py::none();
}
}  // namespace

PYBIND11_DECLARE_HOLDER_TYPE(T, ir::IntrusivePtr<T>, true);
PYBIND11_MODULE(_dairpy, m) {
  m.doc() = "Python binding for DA IR";

  py::enum_<ops::Op>(m, "Op")
#define OP(O) .value(#O, ops::Op_##O)
#include "ops/op_def/ops.list"
#undef OP
    .export_values();

  py::class_<ir::Value, ir::ValuePtr>(m, "Value")
    .def("update_tensor_data", [](ir::Value &self, const at::Tensor &atTensor) {
      ir::DataType type = FromTorchDType(atTensor.scalar_type());
      std::vector<int64_t> shape(atTensor.sizes().begin(), atTensor.sizes().end());
      void *data = atTensor.data_ptr();

      auto tensor = self.ToTensor();
      if (tensor->GetDevice() != FromTorchDevice(atTensor.device())) {
        LOG_EXCEPTION << "Device mismatch in update_tensor_data";
      }

      tensor->SetDtype(type);
      tensor->SetShape(std::move(shape));
      tensor->ResizeStorage();
      tensor->UpdateData(data);
    });

  py::class_<ir::Node, ir::NodePtr>(m, "Node").def_property_readonly(
    "output", [](const ir::NodePtr &node) { return node->output; });

  m.def("from_python", &from_python, py::arg("obj"));
  m.def("to_python", &to_python, py::arg("value"));

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
