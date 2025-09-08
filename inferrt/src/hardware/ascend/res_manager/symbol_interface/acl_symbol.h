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
#ifndef INFERRT_SRC_HARDWARE_ASCEND_ACL_SYMBOL_H_
#define INFERRT_SRC_HARDWARE_ASCEND_ACL_SYMBOL_H_
#include <string>
#include "acl/acl_rt_allocator.h"
#include "hardware/hardware_abstract/dlopen_macro.h"
#include "hardware/ascend/res_manager/symbol_interface/acl_base_symbol.h"

namespace mrt::device::ascend {

ORIGIN_METHOD_WITH_SIMU(aclInit, aclError, const char *);
ORIGIN_METHOD_WITH_SIMU(aclFinalize, aclError);

extern aclInitFunObj aclInit_;
extern aclFinalizeFunObj aclFinalize_;

void LoadAclApiSymbol(const std::string &ascendPath);
void LoadSimulationAclApi();
}  // namespace mrt::device::ascend

#endif  // INFERRT_SRC_HARDWARE_ASCEND_ACL_SYMBOL_H_
