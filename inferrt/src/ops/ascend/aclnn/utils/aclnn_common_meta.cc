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

#include "ops/ascend/aclnn/utils/aclnn_common_meta.h"

namespace mrt {
namespace ops {
DECLARE_ACLNN_COMMON_META_FUNC(aclCreateTensor);
DECLARE_ACLNN_COMMON_META_FUNC(aclCreateScalar);
DECLARE_ACLNN_COMMON_META_FUNC(aclCreateIntArray);
DECLARE_ACLNN_COMMON_META_FUNC(aclCreateFloatArray);
DECLARE_ACLNN_COMMON_META_FUNC(aclCreateBoolArray);
DECLARE_ACLNN_COMMON_META_FUNC(aclCreateTensorList);

DECLARE_ACLNN_COMMON_META_FUNC(aclDestroyTensor);
DECLARE_ACLNN_COMMON_META_FUNC(aclDestroyScalar);
DECLARE_ACLNN_COMMON_META_FUNC(aclDestroyIntArray);
DECLARE_ACLNN_COMMON_META_FUNC(aclDestroyFloatArray);
DECLARE_ACLNN_COMMON_META_FUNC(aclDestroyBoolArray);
DECLARE_ACLNN_COMMON_META_FUNC(aclDestroyTensorList);
DECLARE_ACLNN_COMMON_META_FUNC(aclDestroyAclOpExecutor);

DECLARE_ACLNN_COMMON_META_FUNC(aclnnInit);
DECLARE_ACLNN_COMMON_META_FUNC(aclnnFinalize);

DECLARE_ACLNN_COMMON_META_FUNC(aclSetAclOpExecutorRepeatable);

DECLARE_ACLNN_COMMON_META_FUNC(aclSetTensorAddr);
DECLARE_ACLNN_COMMON_META_FUNC(aclSetDynamicTensorAddr);

}  // namespace ops
}  // namespace mrt
