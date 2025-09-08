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
#ifndef INFERRT_SRC_HARDWARE_ASCEND_ACL_TDT_SYMBOL_H_
#define INFERRT_SRC_HARDWARE_ASCEND_ACL_TDT_SYMBOL_H_
#include <cstddef>
#include <string>
#include "acl/acl_tdt.h"
#include "hardware/hardware_abstract/dlopen_macro.h"

namespace mrt::device::ascend {

ORIGIN_METHOD_WITH_SIMU(acltdtAddDataItem, aclError, acltdtDataset *, acltdtDataItem *)
ORIGIN_METHOD_WITH_SIMU(acltdtCleanChannel, aclError, acltdtChannelHandle *)
ORIGIN_METHOD_WITH_SIMU(acltdtCreateChannel, acltdtChannelHandle *, uint32_t, const char *)
ORIGIN_METHOD_WITH_SIMU(acltdtCreateChannelWithCapacity, acltdtChannelHandle *, uint32_t, const char *, size_t)
ORIGIN_METHOD_WITH_SIMU(acltdtCreateDataItem, acltdtDataItem *, acltdtTensorType, const int64_t *, size_t, aclDataType,
                        void *, size_t)
ORIGIN_METHOD_WITH_SIMU(acltdtCreateDataset, acltdtDataset *)
ORIGIN_METHOD_WITH_SIMU(acltdtDestroyChannel, aclError, acltdtChannelHandle *)
ORIGIN_METHOD_WITH_SIMU(acltdtDestroyDataItem, aclError, acltdtDataItem *)
ORIGIN_METHOD_WITH_SIMU(acltdtDestroyDataset, aclError, acltdtDataset *)
ORIGIN_METHOD_WITH_SIMU(acltdtGetDataAddrFromItem, void *, const acltdtDataItem *)
ORIGIN_METHOD_WITH_SIMU(acltdtGetDataItem, acltdtDataItem *, const acltdtDataset *, size_t)
ORIGIN_METHOD_WITH_SIMU(acltdtGetDatasetName, const char *, const acltdtDataset *)
ORIGIN_METHOD_WITH_SIMU(acltdtGetDatasetSize, size_t, const acltdtDataset *)
ORIGIN_METHOD_WITH_SIMU(acltdtGetDataSizeFromItem, size_t, const acltdtDataItem *)
ORIGIN_METHOD_WITH_SIMU(acltdtGetDataTypeFromItem, aclDataType, const acltdtDataItem *)
ORIGIN_METHOD_WITH_SIMU(acltdtGetDimNumFromItem, size_t, const acltdtDataItem *)
ORIGIN_METHOD_WITH_SIMU(acltdtGetDimsFromItem, aclError, const acltdtDataItem *, int64_t *, size_t)
ORIGIN_METHOD_WITH_SIMU(acltdtGetTensorTypeFromItem, acltdtTensorType, const acltdtDataItem *)
ORIGIN_METHOD_WITH_SIMU(acltdtGetSliceInfoFromItem, aclError, const acltdtDataItem *, size_t *, size_t *)
ORIGIN_METHOD_WITH_SIMU(acltdtQueryChannelSize, aclError, const acltdtChannelHandle *, size_t *)
ORIGIN_METHOD_WITH_SIMU(acltdtReceiveTensor, aclError, const acltdtChannelHandle *, acltdtDataset *, int32_t)
ORIGIN_METHOD_WITH_SIMU(acltdtSendTensor, aclError, const acltdtChannelHandle *, const acltdtDataset *, int32_t)
ORIGIN_METHOD_WITH_SIMU(acltdtStopChannel, aclError, acltdtChannelHandle *)

void LoadAcltdtApiSymbol(const std::string &ascendPath);
void LoadSimulationTdtApi();
}  // namespace mrt::device::ascend

#endif  // INFERRT_SRC_HARDWARE_ASCEND_ACL_TDT_SYMBOL_H_
