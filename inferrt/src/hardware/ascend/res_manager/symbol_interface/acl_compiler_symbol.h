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
#ifndef INFERRT_SRC_HARDWARE_ASCEND_ACL_COMPILER_SYMBOL_H_
#define INFERRT_SRC_HARDWARE_ASCEND_ACL_COMPILER_SYMBOL_H_
#include <string>
#include "acl/acl_op_compiler.h"
#include "hardware/hardware_abstract/dlopen_macro.h"

namespace mrt::device::ascend {

ORIGIN_METHOD_WITH_SIMU(aclopCompileAndExecute, aclError, const char *, int, const aclTensorDesc *const[],
                        const aclDataBuffer *const[], int, const aclTensorDesc *const[], aclDataBuffer *const[],
                        const aclopAttr *, aclopEngineType, aclopCompileType, const char *, aclrtStream);
ORIGIN_METHOD_WITH_SIMU(aclopCompileAndExecuteV2, aclError, const char *, int, aclTensorDesc *[], aclDataBuffer *[],
                        int, aclTensorDesc *[], aclDataBuffer *[], aclopAttr *, aclopEngineType, aclopCompileType,
                        const char *, aclrtStream);
ORIGIN_METHOD_WITH_SIMU(aclSetCompileopt, aclError, aclCompileOpt, const char *);
ORIGIN_METHOD_WITH_SIMU(aclopSetCompileFlag, aclError, aclOpCompileFlag);
ORIGIN_METHOD_WITH_SIMU(aclGenGraphAndDumpForOp, aclError, const char *, int, const aclTensorDesc *const[],
                        const aclDataBuffer *const[], int, const aclTensorDesc *const[], aclDataBuffer *const[],
                        const aclopAttr *, aclopEngineType, const char *, const aclGraphDumpOption *);

void LoadAclOpCompilerApiSymbol(const std::string &ascend_path);
void LoadSimulationAclOpCompilerApi();
}  // namespace mrt::device::ascend

#endif  // INFERRT_SRC_HARDWARE_ASCEND_ACL_COMPILER_SYMBOL_H_
