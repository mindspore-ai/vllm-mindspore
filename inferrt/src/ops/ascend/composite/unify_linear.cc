/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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

#include <dlfcn.h>
#include "ops/ascend/composite/unify_linear.h"
#include "ops/op_register.h"
#include "hardware/hardware_abstract/device_context.h"
#include "hardware/hardware_abstract/device_context_manager.h"
#include "hardware/hardware_abstract/collective/collective_manager.h"
#include "hardware/ascend/res_manager/symbol_interface/symbol_utils.h"
#include "common/logger.h"

namespace mrt {
namespace ops {

UnifyLinear::UnifyLinear() : use_atb_linear_(false), atb_loaded_(false), atb_handle_(nullptr) {
  auto soc = mrt::device::ascend::GetAscendSocVersion();
  if (soc != nullptr) {
    const std::string socName(soc);
    LOG_OUT << "Soc version: " << socName;
    if (socName.rfind("Ascend310", 0) == 0 || socName.rfind("ASCEND310", 0) == 0) {
      use_atb_linear_ = true;
    }
  }
  linear_op_ = CreateLinearOperator();
}

std::unique_ptr<Operator> UnifyLinear::CreateLinearOperator() {
  if (use_atb_linear_) {
    std::stringstream errMsg;
    if (lib_loader_.LoadDynamicLib("libops_ascend_atb.so", &errMsg)) {
      atb_handle_ = lib_loader_.GetHandle("libops_ascend_atb.so");

      if (atb_handle_ != nullptr) {
        typedef void *(*CreateAtbLinearFunc)();
        CreateAtbLinearFunc create_func = reinterpret_cast<CreateAtbLinearFunc>(dlsym(atb_handle_, "CreateAtbLinear"));

        if (create_func != nullptr) {
          void *atb_linear_ptr = create_func();
          if (atb_linear_ptr != nullptr) {
            atb_loaded_ = true;
            LOG_OUT << "Device is Ascend 310 series, successfully loaded AtbLinear operator.";
            return std::unique_ptr<Operator>(static_cast<Operator *>(atb_linear_ptr));
          }
        }
      }
    }

    LOG_OUT << "Failed to load AtbLinear for Ascend 310, falling back to AclnnLinear. Error: " << errMsg.str();
  }

  LOG_OUT << "Using AclnnLinear operator.";
  return std::make_unique<AclnnLinear>();
}

OpsErrorCode UnifyLinear::CalcWorkspace(const std::vector<const ir::Value *> &input, const ir::Value *output,
                                        size_t *workspaceSize) {
  CHECK_IF_NULL(linear_op_);
  return linear_op_->CalcWorkspace(input, output, workspaceSize);
}

OpsErrorCode UnifyLinear::Launch(const std::vector<const ir::Value *> &input, void *workspace, size_t workspaceSize,
                                 ir::Value *output, void *stream) {
  CHECK_IF_NULL(linear_op_);
  return linear_op_->Launch(input, workspace, workspaceSize, output, stream);
}

MRT_REG_OP(linear, UnifyLinear, Ascend);

}  // namespace ops
}  // namespace mrt
