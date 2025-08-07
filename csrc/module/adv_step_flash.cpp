/**
 * Copyright 2025 Huawei Technologies Co., Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <map>
#include <memory>
#include <string>

#include "ms_extension/api.h"

#include "ascendc/adv_step_flash.h"
#include "module/module.h"

struct DtypeCaster {
  ms::Tensor CheckAndCast(const ms::Tensor &t, const std::string &name = "") {
    if (t.data_type() != ms::TypeId::kNumberTypeInt32) {
      if (!name.empty()) {
        tensor_map_[name] = t;
      }
      return t.cast(ms::TypeId::kNumberTypeInt32);
    }
    return t;
  }

  ms::Tensor RecoveryTensorDtype(const ms::Tensor &t, const std::string &name) {
    auto iter = tensor_map_.find(name);
    if (iter == tensor_map_.end()) {
      return t;
    }
    auto ori_tensor = iter->second;
    auto ret = t.cast(ori_tensor.data_type());
    ori_tensor.AssignTensor(ret);
    return ori_tensor;
  }
  std::map<std::string, ms::Tensor> tensor_map_;
};

class AdvStepFlashOp : public ms::pynative::PyboostRunner {
public:
  using PyboostRunner::PyboostRunner;
  void LaunchKernel() override {
    uint8_t *sampledTokenIdsPtr =
        static_cast<uint8_t *>(inputs()[0].GetDataPtr());
    uint8_t *seqLensPtr = static_cast<uint8_t *>(inputs()[1].GetDataPtr());
    uint8_t *blockTablesPtr = static_cast<uint8_t *>(inputs()[2].GetDataPtr());
    uint8_t *inputTokensPtr = static_cast<uint8_t *>(outputs()[0].GetDataPtr());
    uint8_t *inputPositionsPtr =
        static_cast<uint8_t *>(outputs()[1].GetDataPtr());
    uint8_t *slotMappingPtr = static_cast<uint8_t *>(outputs()[3].GetDataPtr());
    auto stride = inputs()[2].stride();
    int32_t block_tables_stride = stride.empty() ? 1 : stride[0];

    uint32_t blockDims = 1;
    void *l2ctrl = nullptr;
    AdvStepFlashKernelEntry(blockDims, l2ctrl, stream(), sampledTokenIdsPtr,
                            blockTablesPtr, seqLensPtr, inputTokensPtr,
                            inputPositionsPtr, seqLensPtr, slotMappingPtr,
                            num_seqs_, block_size_, block_tables_stride);
  }

  static void Eval(int32_t num_seqs, int32_t num_queries, int32_t block_size,
                   ms::Tensor input_tokens,      // output
                   ms::Tensor sampled_token_ids, // input
                   ms::Tensor input_positions,   // output
                   ms::Tensor seq_lens,          // input&output (inplace)
                   ms::Tensor slot_mapping,      // output
                   ms::Tensor block_tables       // input
  ) {
    // the AdvStepFlashKernelEntry only support int32 inputs.
    DtypeCaster caster;
    sampled_token_ids = caster.CheckAndCast(sampled_token_ids);
    block_tables = caster.CheckAndCast(block_tables);
    input_tokens = caster.CheckAndCast(input_tokens, "input_tokens");
    input_positions = caster.CheckAndCast(input_positions, "input_positions");
    slot_mapping = caster.CheckAndCast(slot_mapping, "slot_mapping");
    seq_lens = caster.CheckAndCast(seq_lens, "seq_lens");

    auto runner = std::make_shared<AdvStepFlashOp>("AdvanceStepFlashattn");
    runner->num_seqs_ = num_seqs;
    runner->num_queries_ = num_queries;
    runner->block_size_ = block_size;
    runner->Run({sampled_token_ids, seq_lens, block_tables},
                {input_tokens, input_positions, seq_lens, slot_mapping});

    input_tokens = caster.RecoveryTensorDtype(input_tokens, "input_tokens");
    input_positions =
        caster.RecoveryTensorDtype(input_positions, "input_positions");
    slot_mapping = caster.RecoveryTensorDtype(slot_mapping, "slot_mapping");
    seq_lens = caster.RecoveryTensorDtype(seq_lens, "seq_lens");
  }
  int32_t num_seqs_{0};
  int32_t num_queries_{0};
  int32_t block_size_{0};
};

auto pyboost_adv_step_flash(int32_t num_seqs, int32_t num_queries,
                            int32_t block_size, ms::Tensor input_tokens,
                            ms::Tensor sampled_token_ids,
                            ms::Tensor input_positions, ms::Tensor seq_lens,
                            ms::Tensor slot_mapping, ms::Tensor block_tables) {
  return ms::pynative::PyboostRunner::Call<0>(
      AdvStepFlashOp::Eval, num_seqs, num_queries, block_size, input_tokens,
      sampled_token_ids, input_positions, seq_lens, slot_mapping, block_tables);
}

VLLM_MS_EXTENSION_MODULE(m) {
  m.def("advance_step_flashattn", &pyboost_adv_step_flash,
        "advance_step_flashattn", pybind11::arg("num_seqs"),
        pybind11::arg("num_queries"), pybind11::arg("block_size"),
        pybind11::arg("input_tokens"), pybind11::arg("sampled_token_ids"),
        pybind11::arg("input_positions"), pybind11::arg("seq_lens"),
        pybind11::arg("slot_mapping"), pybind11::arg("block_tables"));
}
