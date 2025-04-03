#include <string>
#include <map>
#include <memory>

#include "ms_extension.h"

#include "adv_step_flash.h"

using BaseTensor = mindspore::tensor::BaseTensor;
using BaseTensorPtr = mindspore::tensor::BaseTensorPtr;
using PyBoostUtils = mindspore::kernel::pyboost::PyBoostUtils;

uint8_t *GetDataPtr(const BaseTensorPtr &t) {
  return static_cast<uint8_t *>(t->device_address()->GetMutablePtr()) + t->data().itemsize() * t->storage_offset();
}

struct DtypeCaster {
  BaseTensorPtr CheckAndCast(const BaseTensorPtr &t, const std::string &name = "") {
    mindspore::Int64ImmPtr dst_type = std::make_shared<mindspore::Int64Imm>(mindspore::TypeId::kNumberTypeInt32);
    if (t->data_type() != mindspore::TypeId::kNumberTypeInt32) {
      if (!name.empty()) {
        tensor_map_[name] = t;
      }
      return mindspore::kernel::pyboost::cast(t, dst_type);
    }
    return t;
  }
  BaseTensorPtr RecoveryTensorDtype(const BaseTensorPtr &t, const std::string &name) {
    auto iter = tensor_map_.find(name);
    if (iter == tensor_map_.end()) {
      return t;
    }
    auto ori_tensor = iter->second;
    auto ori_dtype = std::make_shared<mindspore::Int64Imm>(ori_tensor->data_type());
    auto ret = mindspore::kernel::pyboost::cast(t, ori_dtype);
    ori_tensor->AssignValue(*ret);
    return ori_tensor;
  }
  std::map<std::string, BaseTensorPtr> tensor_map_;
};

void AdvStepFlashAscendC(int32_t num_seqs, int32_t num_queries, int32_t block_size,
                         BaseTensorPtr &input_tokens,      // output
                         BaseTensorPtr sampled_token_ids,  // input
                         BaseTensorPtr &input_positions,   // output
                         BaseTensorPtr &seq_lens,          // input&output (inplace)
                         BaseTensorPtr &slot_mapping,      // output
                         BaseTensorPtr block_tables        // input
) {
  // the AdvStepFlashKernelEntry only support int32 inputs.
  DtypeCaster caster;
  sampled_token_ids = caster.CheckAndCast(sampled_token_ids);
  block_tables = caster.CheckAndCast(block_tables);
  input_tokens = caster.CheckAndCast(input_tokens, "input_tokens");
  input_positions = caster.CheckAndCast(input_positions, "input_positions");
  slot_mapping = caster.CheckAndCast(slot_mapping, "slot_mapping");
  seq_lens = caster.CheckAndCast(seq_lens, "seq_lens");

  auto stream_id = PyBoostUtils::cur_stream_id();
  auto device_context = mindspore::runtime::OpRunner::GetDeviceContext("Ascend");
  PyBoostUtils::PrepareOpInputs(device_context, stream_id, input_tokens, sampled_token_ids, input_positions, seq_lens,
                                slot_mapping, block_tables);
  // PyBoostUtils::PrepareOpOutputs(device_context, stream_id, outputs);
  PyBoostUtils::DispatchRun(std::make_shared<mindspore::runtime::PyBoostDeviceTask>([=]() {
    PyBoostUtils::MallocOpInputs(device_context, input_tokens, sampled_token_ids, input_positions, seq_lens,
                                 slot_mapping, block_tables);
    // PyBoostUtils::MallocOpOutputs(device_context, outputs);

    uint8_t *sampledTokenIdsPtr = GetDataPtr(sampled_token_ids);
    uint8_t *blockTablesPtr = GetDataPtr(block_tables);
    uint8_t *seqLensPtr = GetDataPtr(seq_lens);
    uint8_t *inputTokensPtr = GetDataPtr(input_tokens);
    uint8_t *inputPositionsPtr = GetDataPtr(input_positions);
    uint8_t *slotMappingPtr = GetDataPtr(slot_mapping);
    auto aclStream = device_context->device_res_manager_->GetStream(stream_id);
    auto stride = block_tables->stride();
    int32_t block_tables_stride = stride.empty() ? 1 : stride[0];

    mindspore::runtime::OpExecutor::DispatchLaunchTask([=]() {
      uint32_t blockDims = 1;
      void *l2ctrl = nullptr;
      AdvStepFlashKernelEntry(blockDims, l2ctrl, aclStream, sampledTokenIdsPtr, blockTablesPtr, seqLensPtr,
                              inputTokensPtr, inputPositionsPtr, seqLensPtr, slotMappingPtr, num_seqs, block_size,
                              block_tables_stride);
    });
  }));

  input_tokens = caster.RecoveryTensorDtype(input_tokens, "input_tokens");
  input_positions = caster.RecoveryTensorDtype(input_positions, "input_positions");
  slot_mapping = caster.RecoveryTensorDtype(slot_mapping, "slot_mapping");
  seq_lens = caster.RecoveryTensorDtype(seq_lens, "seq_lens");
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("adv_step_flash", &AdvStepFlashAscendC, "adv_step_flash_ascendc", pybind11::arg("num_seqs"),
        pybind11::arg("num_queries"), pybind11::arg("block_size"), pybind11::arg("input_tokens"),
        pybind11::arg("sampled_token_ids"), pybind11::arg("input_positions"), pybind11::arg("seq_lens"),
        pybind11::arg("slot_mapping"), pybind11::arg("block_tables"));
}
