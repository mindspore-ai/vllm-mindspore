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
#ifndef INFERRT_SRC_HARDWARE_ASCEND_ACL_RT_ALLOCATOR_SYMBOL_H_
#define INFERRT_SRC_HARDWARE_ASCEND_ACL_RT_ALLOCATOR_SYMBOL_H_
#include <string>
#include "acl/acl_rt_allocator.h"
#include "hardware/hardware_abstract/dlopen_macro.h"

namespace mrt::device::ascend {
ORIGIN_METHOD_WITH_SIMU(aclrtAllocatorCreateDesc, aclrtAllocatorDesc)
ORIGIN_METHOD_WITH_SIMU(aclrtAllocatorDestroyDesc, aclError, aclrtAllocatorDesc)
ORIGIN_METHOD_WITH_SIMU(aclrtAllocatorRegister, aclError, aclrtStream, aclrtAllocatorDesc)
ORIGIN_METHOD_WITH_SIMU(aclrtAllocatorSetAllocAdviseFuncToDesc, aclError, aclrtAllocatorDesc,
                        aclrtAllocatorAllocAdviseFunc)
ORIGIN_METHOD_WITH_SIMU(aclrtAllocatorSetAllocFuncToDesc, aclError, aclrtAllocatorDesc, aclrtAllocatorAllocFunc)
ORIGIN_METHOD_WITH_SIMU(aclrtAllocatorSetFreeFuncToDesc, aclError, aclrtAllocatorDesc, aclrtAllocatorFreeFunc)
ORIGIN_METHOD_WITH_SIMU(aclrtAllocatorSetGetAddrFromBlockFuncToDesc, aclError, aclrtAllocatorDesc,
                        aclrtAllocatorGetAddrFromBlockFunc)
ORIGIN_METHOD_WITH_SIMU(aclrtAllocatorSetObjToDesc, aclError, aclrtAllocatorDesc, aclrtAllocator)
ORIGIN_METHOD_WITH_SIMU(aclrtAllocatorUnregister, aclError, aclrtStream)

extern aclrtAllocatorCreateDescFunObj aclrtAllocatorCreateDesc_;
extern aclrtAllocatorDestroyDescFunObj aclrtAllocatorDestroyDesc_;
extern aclrtAllocatorRegisterFunObj aclrtAllocatorRegister_;
extern aclrtAllocatorSetAllocAdviseFuncToDescFunObj aclrtAllocatorSetAllocAdviseFuncToDesc_;
extern aclrtAllocatorSetAllocFuncToDescFunObj aclrtAllocatorSetAllocFuncToDesc_;
extern aclrtAllocatorSetFreeFuncToDescFunObj aclrtAllocatorSetFreeFuncToDesc_;
extern aclrtAllocatorSetGetAddrFromBlockFuncToDescFunObj aclrtAllocatorSetGetAddrFromBlockFuncToDesc_;
extern aclrtAllocatorSetObjToDescFunObj aclrtAllocatorSetObjToDesc_;
extern aclrtAllocatorUnregisterFunObj aclrtAllocatorUnregister_;

void LoadAclAllocatorApiSymbol(const std::string &ascend_path);
void LoadSimulationAclAllocatorApi();
}  // namespace mrt::device::ascend

#endif  // INFERRT_SRC_HARDWARE_ASCEND_ACL_RT_ALLOCATOR_SYMBOL_H_
