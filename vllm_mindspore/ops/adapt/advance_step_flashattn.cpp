#include <iostream>
#include <vector>
#include <pybind11/pybind11.h>
#include "ir/tensor.h"
#include "acl/acl.h"

#include "ms_extension.h"
#include "mindspore/ccsrc/pyboost/pyboost_utils.h"
#include "mindspore/ccsrc/pyboost/op_runner.h"
#include "runtime/pynative/op_runner.h"
#include "runtime/pynative/op_executor.h"
#if __has_include("mindspore/ccsrc/runtime/runtime_conf/runtime_conf.h")
#include "mindspore/ccsrc/runtime/runtime_conf/runtime_conf.h"
#else
#include "mindspore/ccsrc/include/common/runtime_conf/runtime_conf.h"
#endif
#if __has_include("mindspore/ccsrc/runtime/pynative/task/device_task.h")
#include "mindspore/ccsrc/runtime/pynative/task/device_task.h"
#else
#include "mindspore/ccsrc/runtime/pipeline/task/device_task.h"
#endif
#include "mindspore/ccsrc/plugin/res_manager/ascend/stream_manager/ascend_stream_manager.h"

#include "ascendc/adv_step_flash.h"

using BaseTensor = mindspore::tensor::BaseTensor;
using BaseTensorPtr = mindspore::tensor::BaseTensorPtr;

uint8_t *GetDataPtr(const BaseTensorPtr &t) {
  return static_cast<uint8_t *>(t->device_address()->GetMutablePtr()) + t->data().itemsize() * t->storage_offset();
}

using PyBoostUtils = mindspore::kernel::pyboost::PyBoostUtils;

void AdvStepFlashAscendC(int32_t num_seqs,
                         int32_t num_queries,
                         int32_t block_size,
                         const BaseTensorPtr &input_tokens,
                         const BaseTensorPtr &sampled_token_ids,
                         const BaseTensorPtr &input_positions,
                         const BaseTensorPtr &seq_lens,
                         const BaseTensorPtr &slot_mapping,
                         const BaseTensorPtr &block_tables) {
  auto stream_id = PyBoostUtils::cur_stream_id();
  auto device_context = mindspore::runtime::OpRunner::GetDeviceContext("Ascend");
  PyBoostUtils::PrepareOpInputs(device_context, stream_id, input_tokens, sampled_token_ids, input_positions, seq_lens,
                                  slot_mapping, block_tables);
  //   PyBoostUtils::PrepareOpOutputs(device_context, stream_id, outputs);
  PyBoostUtils::DispatchRun(std::make_shared<mindspore::runtime::PyBoostDeviceTask>([=]() {
    PyBoostUtils::MallocOpInputs(device_context, input_tokens, sampled_token_ids, input_positions, seq_lens,
                                  slot_mapping, block_tables);
    //   PyBoostUtils::MallocOpOutputs(device_context, outputs);

    uint8_t *sampledTokenIdsPtr = GetDataPtr(sampled_token_ids);
    uint8_t *blockTablesPtr = GetDataPtr(block_tables);
    uint8_t *seqLensPtr = GetDataPtr(seq_lens);
    uint8_t *inputTokensPtr = GetDataPtr(input_tokens);
    uint8_t *inputPositionsPtr = GetDataPtr(input_positions);
    uint8_t *slotMappingPtr = GetDataPtr(slot_mapping);
    auto aclStream = device_context->device_res_manager_->GetStream(stream_id);
    auto stride = block_tables->stride();
    int32_t block_tables_stride = stride.empty() ? 1: stride[0];

    mindspore::runtime::OpExecutor::DispatchLaunchTask([=]() {
      uint32_t blockDims = 1;
      void *l2ctrl = nullptr;
      AdvStepFlashKernelEntry(blockDims, l2ctrl, aclStream, sampledTokenIdsPtr, blockTablesPtr, seqLensPtr,
                              inputTokensPtr, inputPositionsPtr, seqLensPtr, slotMappingPtr, num_seqs, block_size,
                              block_tables_stride);
    });
  }));
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("adv_step_flash", &AdvStepFlashAscendC, "adv_step_flash_ascendc", pybind11::arg("num_seqs"),
    pybind11::arg("num_queries"), pybind11::arg("block_size"), pybind11::arg("input_tokens"),
    pybind11::arg("sampled_token_ids"), pybind11::arg("input_positions"), pybind11::arg("seq_lens"),
    pybind11::arg("slot_mapping"), pybind11::arg("block_tables"));
}
