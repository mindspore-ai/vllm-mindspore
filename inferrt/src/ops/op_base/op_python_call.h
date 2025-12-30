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

#ifndef __OPS_OP_BASE_OP_PYTHON_CALL_H__
#define __OPS_OP_BASE_OP_PYTHON_CALL_H__

#include <pybind11/pybind11.h>
#include <string>
#include <vector>
#include "ops/op_register.h"

namespace mrt {
namespace ops {
namespace py = pybind11;
class OpPythonCall : public Operator {
 public:
  OpPythonCall() = default;
  ~OpPythonCall() override = default;

  void Init(const std::vector<const ir::Value *> &inputs, const ir::Value *output);

  OpsErrorCode Launch(const std::vector<const ir::Value *> &input, void *workspace, size_t workspaceSize,
                      ir::Value *output, void *stream);

 protected:
  py::tuple PreprocessInputs(const std::vector<const ir::Value *> &input);
  OpsErrorCode PostprocessOutputs(py::handle result, ir::Value *output, void *stream);
  std::string opName_;
  std::string moduleName_;
  std::vector<const ir::Value *> inputs_;
  py::function pyFunc_;
};
}  // namespace ops
}  // namespace mrt

#endif  // __OPS_OP_BASE_OP_CUSTOM_CALL_H__
