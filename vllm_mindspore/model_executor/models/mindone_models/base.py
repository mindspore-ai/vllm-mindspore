# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from vllm.config import VllmConfig

from vllm_mindspore.model_executor.models.model_base import MsModelBase


class MindONEModelBase(MsModelBase):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)

    def named_parameters(self):
        self._check_modules_valid()

        for cell_name, module in self.modules_dict.items():
            if isinstance(module, MindONEModelBase):
                for cell_name, sub_module in module.modules_dict.items():
                    for par_name, par in sub_module.parameters_and_names():
                        if cell_name != "self":
                            par_name = cell_name + "." + par_name

                        yield par_name, par
            else:
                for par_name, par in module.parameters_and_names():
                    if cell_name != "self":
                        par_name = cell_name + "." + par_name

                    yield par_name, par

    def get_params_dict(self):
        self._check_modules_valid()

        params_dict = dict()
        for name, module in self.modules_dict.items():
            if isinstance(module, MindONEModelBase):
                params_dict.update(module.get_params_dict())
            else:
                module_params = module.parameters_dict()
                if name != "self":
                    new_module_params = dict()
                    for param_name, param in module_params.items():
                        new_module_params[name + "." + param_name] = param
                    module_params = new_module_params
                params_dict.update(module_params)

        return params_dict

    def named_modules(self, remove_duplicate: bool = True):
        self._check_modules_valid()

        for name, module in self.modules_dict.items():
            if isinstance(module, MindONEModelBase):
                for name, sub_module in module.modules_dict.items():
                    for module_name, sub_sub_module in sub_module.cells_and_names(  # noqa: E501
                    ):
                        if name != "self":
                            module_name = name + "." + module_name
                        yield module_name, sub_sub_module
            else:
                for module_name, sub_module in module.cells_and_names():
                    if name != "self":
                        module_name = name + "." + module_name
                    yield module_name, sub_module

    def eval(self):
        self._check_modules_valid()

        for _, module in self.modules_dict.items():
            if isinstance(module, MindONEModelBase):
                module.eval()
            else:
                module.set_train(False)

        return self

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
