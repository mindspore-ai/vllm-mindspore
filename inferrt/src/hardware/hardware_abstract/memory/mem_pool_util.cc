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
#include "hardware/hardware_abstract/memory/mem_pool_util.h"
#include <map>

namespace mrt {
namespace memory {
namespace mem_pool {
const std::map<MemType, std::string> kMemTypeStr = {{MemType::kWeight, "Weight"},
                                                    {MemType::kConstantValue, "ConstantValue"},
                                                    {MemType::kKernel, "Kernel"},
                                                    {MemType::kGraphOutput, "GraphOutput"},
                                                    {MemType::kSomas, "Somas"},
                                                    {MemType::kSomasOutput, "SomasOutput"},
                                                    {MemType::kGeConst, "GeConst"},
                                                    {MemType::kGeFixed, "GeFixed"},
                                                    {MemType::kBatchMemory, "BatchMemory"},
                                                    {MemType::kContinuousMemory, "ContinuousMemory"},
                                                    {MemType::kPyNativeInput, "PyNativeInput"},
                                                    {MemType::kPyNativeOutput, "PyNativeOutput"},
                                                    {MemType::kWorkSpace, "WorkSpace"},
                                                    {MemType::kOther, "Other"}};

std::string MemTypeToStr(MemType mem_type) { return kMemTypeStr.at(mem_type); }
}  // namespace mem_pool
}  // namespace memory
}  // namespace mrt
