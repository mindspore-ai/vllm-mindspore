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

#ifndef __OPS_ASCEND_ATB_ATB_KERNEL_MOD_H__
#define __OPS_ASCEND_ATB_ATB_KERNEL_MOD_H__

#include <vector>
#include <string>
#include <unordered_map>
#include "ops/operator.h"
#include "atb/atb_infer.h"
#include "ops/ascend/atb/atb_adapter.h"

namespace mrt {
namespace ops {

struct AtbCacheEntry {
  atb::Operation *op = nullptr;
  size_t workspace_size = 0;
  bool workspace_cached = false;
};

class AtbBase : public Operator {
 public:
  explicit AtbBase(const std::string &op_name) : op_name_(op_name), current_hash_id_(0), op_(nullptr) {}
  ~AtbBase() override;

  OpsErrorCode CalcWorkspace(const std::vector<const ir::Value *> &inputs, const ir::Value *output,
                             size_t *workspace_size) override = 0;

  OpsErrorCode Launch(const std::vector<const ir::Value *> &inputs, void *workspace, size_t workspaceSize,
                      ir::Value *output, void *stream) override = 0;

 protected:
  template <typename ParamType>
  AtbCacheEntry &GetOrCreateEntry(const ParamType &param, const std::vector<const ir::Value *> &inputs,
                                  const ir::Value *output) {
    uint64_t hash = AtbHash(inputs, output, op_name_);
    current_hash_id_ = hash;

    auto &entry = cache_[hash];
    if (entry.op == nullptr) {
      auto ret = atb::CreateOperation(param, &entry.op);
      if (ret != 0) {
        LOG_EXCEPTION << "Failed to create ATB operation " << op_name_ << ", ret: " << ret;
      }
    }
    op_ = entry.op;
    return entry;
  }

  OpsErrorCode GetWorkspaceSize(AtbCacheEntry &entry, atb::VariantPack &variant_pack, size_t *workspace_size);

  OpsErrorCode LaunchAtb(atb::VariantPack variant_pack, void *workspace_ptr, size_t workspace_size, aclrtStream stream);

  std::string op_name_;
  uint64_t current_hash_id_;
  atb::Operation *op_;
  std::unordered_map<uint64_t, AtbCacheEntry> cache_;
  ParamSetter param_setter_;
};

}  // namespace ops
}  // namespace mrt

#endif
