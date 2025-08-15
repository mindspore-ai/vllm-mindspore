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

#include "runtime/executor.h"
#include "tensor/tensor.h"
#include "ops/ops_name.h"

namespace py = pybind11;
using namespace da;
using namespace da::tensor;
using namespace da::ops;
using namespace da::runtime;

static void SetBufferInfoToDATensor(const py::buffer_info &info, DATensor *tensor) {
  Type type;
  if (info.format == py::format_descriptor<float>::format())
    type = Type_F32;
  else if (info.format == py::format_descriptor<double>::format())
    type = Type_F64;
  else if (info.format == py::format_descriptor<int32_t>::format())
    type = Type_I32;
  else if (info.format == py::format_descriptor<int64_t>::format())
    type = Type_I64;
  else if (info.format == py::format_descriptor<int16_t>::format())
    type = Type_I16;
  else if (info.format == py::format_descriptor<bool>::format())
    type = Type_Bool;
  else
    throw std::runtime_error("Unsupported numpy dtype for buffer conversion");
  tensor->type = type;

  if (info.ndim > DA_TENSOR_MAX_DIM) {
    throw std::runtime_error("Unsupported tensor dimension > " + std::to_string(DA_TENSOR_MAX_DIM));
  }
  tensor->dim = static_cast<size_t>(info.ndim);
  for (ssize_t i = 0; i < info.ndim; ++i) tensor->shape[i] = info.shape[i];

  tensor->data = info.ptr;
  tensor->tensorType = HOST_TENSOR;  // Assuming host tensor for now, can be extended for device tensors
  tensor->op = Op_End;               // Default operation is no-op
  tensor->inputSize = 0;             // No inputs for constants
}

static py::array ConvertDATensorToNumpyArray(DATensor *tensor) {
  if (tensor->data == nullptr) {
    throw std::runtime_error("Cannot convert DATensor with null data to numpy array");
  }
  std::string format;
  switch (tensor->type) {
    case Type_F32:
      format = py::format_descriptor<float>::format();
      break;
    case Type_I32:
      format = py::format_descriptor<int32_t>::format();
      break;
    case Type_I64:
      format = py::format_descriptor<int64_t>::format();
      break;
    case Type_F64:
      format = py::format_descriptor<double>::format();
      break;
    case Type_Bool:
      format = py::format_descriptor<bool>::format();
      break;
    case Type_I16:
      format = py::format_descriptor<int16_t>::format();
      break;
    default:
      throw std::runtime_error("Unsupported dtype for numpy conversion");
  }
  std::vector<size_t> shape(tensor->shape, tensor->shape + tensor->dim);
  std::vector<size_t> strides(tensor->dim);
  if (tensor->dim > 0) {
    strides[tensor->dim - 1] = DataTypeSize(tensor->type);
    for (int i = tensor->dim - 2; i >= 0; --i) {
      strides[i] = strides[i + 1] * tensor->shape[i + 1];
    }
  }
  return py::array(py::dtype(format), shape, strides, tensor->data);
}

static std::vector<py::array> ConvertDATensorToNumpyArrayList(DATensor *tensor) {
  CHECK_IF_NULL(tensor);
  CHECK_IF_FAIL(tensor->type == da::tensor::Type_Tensor);

  auto **tensorList = reinterpret_cast<da::tensor::DATensor **>(tensor->data);
  CHECK_IF_NULL(tensorList);

  std::vector<py::array> result;
  for (size_t i = 0; i < tensor->shape[0]; ++i) {
    (void)result.emplace_back(ConvertDATensorToNumpyArray(tensorList[i]));
  }
  return result;
}

PYBIND11_MODULE(_dairpy, m) {
  m.doc() = "Python binding for DA IR";

  py::enum_<Op>(m, "Op")
#define OP(O) .value(#O, Op_##O)
#include "ops/ops.list"
#undef OP
    .export_values();

  py::class_<DATensor>(m, "DATensor")
    .def("__repr__",
         [](const DATensor &t) {
           std::stringstream ss;
           ss << "DATensor(op=" << ToStr(t.op) << ", shape=[";
           for (size_t i = 0; i < t.dim; ++i) {
             ss << t.shape[i] << (i == t.dim - 1 ? "" : ", ");
           }
           ss << "])";
           return ss.str();
         })
    .def("update_data",
         [](DATensor &self, py::buffer b) {
           py::buffer_info info = b.request();
           self.data = info.ptr;
         })
    .def("numpy", ConvertDATensorToNumpyArray)
    .def("list", ConvertDATensorToNumpyArrayList);

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
    .def("cast_to_tensor_list", &GraphExecutor::CastToTensorList, py::arg("tensor"), py::arg("len"))
    .def("add_parameter", &GraphExecutor::AddParameter, py::arg("tensor"))
    .def(
      "add_op",
      [](GraphExecutor &self, Op op, const std::vector<DATensor *> &inputs) -> DATensor * {
        return self.AddTensor(op, inputs);
      },
      py::arg("op"), py::arg("inputs"), py::return_value_policy::reference)
    .def(
      "add_const",
      [](GraphExecutor &self, py::buffer b) -> DATensor * {
        py::buffer_info info = b.request();
        auto tensor = self.AddTensor();
        SetBufferInfoToDATensor(info, tensor);
        return tensor;
      },
      py::arg("tensor"), py::return_value_policy::reference);
}
