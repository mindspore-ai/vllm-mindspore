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
#ifndef INFERRT_SRC_HARDWARE_ASCEND_ACL_OP_SYMBOL_H_
#define INFERRT_SRC_HARDWARE_ASCEND_ACL_OP_SYMBOL_H_
#include <string>
#include "acl/acl_op.h"
#include "hardware/hardware_abstract/dlopen_macro.h"

namespace mrt::device::ascend {

ORIGIN_METHOD_WITH_SIMU(aclopCreateAttr, aclopAttr *)
ORIGIN_METHOD_WITH_SIMU(aclopSetAttrBool, aclError, aclopAttr *, const char *, uint8_t)
ORIGIN_METHOD_WITH_SIMU(aclopSetAttrDataType, aclError, aclopAttr *, const char *, aclDataType)
ORIGIN_METHOD_WITH_SIMU(aclopSetAttrFloat, aclError, aclopAttr *, const char *, float)
ORIGIN_METHOD_WITH_SIMU(aclopSetAttrInt, aclError, aclopAttr *, const char *, int64_t)
ORIGIN_METHOD_WITH_SIMU(aclopSetAttrListBool, aclError, aclopAttr *, const char *, int, const uint8_t *)
ORIGIN_METHOD_WITH_SIMU(aclopSetAttrListDataType, aclError, aclopAttr *, const char *, int, const aclDataType[])
ORIGIN_METHOD_WITH_SIMU(aclopSetAttrListFloat, aclError, aclopAttr *, const char *, int, const float *)
ORIGIN_METHOD_WITH_SIMU(aclopSetAttrListInt, aclError, aclopAttr *, const char *, int, const int64_t *)
ORIGIN_METHOD_WITH_SIMU(aclopSetAttrListListInt, aclError, aclopAttr *, const char *, int, const int *,
                        const int64_t *const[])
ORIGIN_METHOD_WITH_SIMU(aclopSetAttrListString, aclError, aclopAttr *, const char *, int, const char **)
ORIGIN_METHOD_WITH_SIMU(aclopSetAttrString, aclError, aclopAttr *, const char *, const char *)
ORIGIN_METHOD_WITH_SIMU(aclopSetModelDir, aclError, const char *)

extern aclopCreateAttrFunObj aclopCreateAttr_;
extern aclopSetAttrBoolFunObj aclopSetAttrBool_;
extern aclopSetAttrDataTypeFunObj aclopSetAttrDataType_;
extern aclopSetAttrFloatFunObj aclopSetAttrFloat_;
extern aclopSetAttrIntFunObj aclopSetAttrInt_;
extern aclopSetAttrListBoolFunObj aclopSetAttrListBool_;
extern aclopSetAttrListDataTypeFunObj aclopSetAttrListDataType_;
extern aclopSetAttrListFloatFunObj aclopSetAttrListFloat_;
extern aclopSetAttrListIntFunObj aclopSetAttrListInt_;
extern aclopSetAttrListListIntFunObj aclopSetAttrListListInt_;
extern aclopSetAttrListStringFunObj aclopSetAttrListString_;
extern aclopSetAttrStringFunObj aclopSetAttrString_;
extern aclopSetModelDirFunObj aclopSetModelDir_;

void LoadAclOpApiSymbol(const std::string &ascendPath);
void LoadSimulationAclOpApi();
}  // namespace mrt::device::ascend

#endif  // INFERRT_SRC_HARDWARE_ASCEND_ACL_OP_SYMBOL_H_
