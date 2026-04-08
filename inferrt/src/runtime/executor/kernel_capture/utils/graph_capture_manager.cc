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

#include "runtime/executor/kernel_capture/utils/graph_capture_manager.h"
#include <algorithm>
#include <sstream>
#include <cctype>
#include "runtime/executor/kernel_capture/kernel_capture_executor.h"
#include "ops/utils/async.h"
#include "ir/graph.h"

namespace mrt {
namespace runtime {

namespace {

constexpr bool kEventDisableTiming = false;
constexpr bool kEventBlockingWait = true;
constexpr bool kEventDisableInterprocess = false;

bool IsCustomCallOp(const OpRunner &opRunner) {
  return opRunner.GetOpType() == ops::OpType::TorchCallOp || opRunner.GetOpType() == ops::OpType::PythonCallOp;
}

}  // namespace

void GraphCaptureManager::Initialize(const std::vector<OpRunner> &opRunners,
                                     const device::DeviceContext *expected_device_context) {
  FindSupportCaptureKernelPositions(opRunners, expected_device_context);
  LOG_OUT << "Single op number : " << singleOpPos_;
  InitializeSingleOpUpdateResources(expected_device_context);
}

void GraphCaptureManager::InitializeSingleOpUpdateResources(const device::DeviceContext *expected_device_context) {
  if (singleOpPos_.empty()) {
    return;
  }

  size_t singleOpSize = singleOpPos_.size();
  singleOpUpdateEvents_.resize(singleOpSize);
  for (size_t i = 0; i < singleOpSize; i++) {
    auto event = expected_device_context->deviceResManager_->CreateEventWithFlag(
      kEventDisableTiming, kEventBlockingWait, kEventDisableInterprocess);
    CHECK_IF_NULL(event);
    singleOpUpdateEvents_[i] = event;
  }
  singleOpUpdateHandles_.resize(singleOpSize);
}

bool GraphCaptureManager::CheckKernelSupportCapture(const OpRunner &opRunner,
                                                    const device::DeviceContext *expected_device_context) {
  // Check if the op runner's device context matches the expected one
  if (hardware::GetDeviceNameByType(opRunner.GetDevice().type) !=
      expected_device_context->GetDeviceContextKey().deviceName_) {
    LOG_EXCEPTION << "Capture graph mode can not support different device kernel: " << opRunner.GetOpName()
                  << ", device type: " << hardware::GetDeviceNameByType(opRunner.GetDevice().type);
    return false;
  }

  // Check if the op needs dynamic shape update (not supported in capture mode)
  // For now, we'll skip this check since the method may not exist
  if (opRunner.GetOperator()->NeedUpdateOutputShapeAfterLaunch()) {
    LOG_EXCEPTION << "Capture graph mode can not support computed depend kernel(whose shape need update after launch): "
                  << opRunner.GetOpName();
    return false;
  }

  // Check if the op is in the skip list (not supported for capture)
  // For now, we'll implement a basic check - in a real implementation,
  // this would come from runtime configuration
  std::string op_name = opRunner.GetOpName();
  std::transform(op_name.begin(), op_name.end(), op_name.begin(), ::tolower);

  // List of ops that should not be captured (example list)
  const std::vector<std::string> &op_capture_skip = KernelCaptureExecutorManager::GetInstance().OpCaptureSkip();

  for (const auto &not_capture_op : op_capture_skip) {
    std::string lower_op = not_capture_op;
    std::transform(lower_op.begin(), lower_op.end(), lower_op.begin(), ::tolower);

    if (op_name == lower_op) {
      LOG_OUT << "Not capturing op: " << not_capture_op;
      return false;
    }
  }

  // Additional checks could be added here based on the op properties
  // For example, check if the op is dynamic, has variable inputs, etc.

  return true;
}

bool GraphCaptureManager::FindSupportCaptureKernelPositions(const std::vector<OpRunner> &opRunners,
                                                            const device::DeviceContext *expected_device_context) {
  if (!capture_kernel_range_positions_.empty()) {
    LOG_ERROR << "GraphCaptureManager has already initialized.";
    return false;
  }
  init_ = true;
  size_t start = 0;
  size_t end = 0;
  bool find_kernel_can_capture = false;
  size_t kernel_num = opRunners.size();
  if (kernel_num < 1) {
    return false;
  }
  for (size_t i = 0; i < kernel_num; i++) {
    const auto &opRunner = opRunners[i];

    if (CheckKernelSupportCapture(opRunner, expected_device_context)) {
      if (!find_kernel_can_capture) {
        start = i;
        end = i;
        find_kernel_can_capture = true;
      } else {
        end = i;
      }
    } else {
      if (find_kernel_can_capture) {
        capture_kernel_range_positions_.emplace_back(start, end);
        executors_.emplace_back(CAPTURE_GRAPH, (capture_kernel_range_positions_.size() - static_cast<size_t>(1)));
      }
      executors_.emplace_back(KERNEL, i);
      singleOpPos_.emplace_back(i);
      find_kernel_can_capture = false;
    }
  }

  if (find_kernel_can_capture) {
    capture_kernel_range_positions_.emplace_back(start, end);
    executors_.emplace_back(CAPTURE_GRAPH, (capture_kernel_range_positions_.size() - static_cast<size_t>(1)));
  }

  capture_graph_num_ = capture_kernel_range_positions_.size();
  LOG_OUT << "Capture graph num: " << capture_graph_num_;

  auto executor_size = executors_.size();
  LOG_OUT << "Dump executor info for capture graph: ";
  for (size_t i = 0; i < executor_size; i++) {
    std::string executor_mode = (executors_[i].first == CAPTURE_GRAPH ? "capture graph" : "kernel");
    std::ostringstream executor_mode_info;
    if (executors_[i].first == CAPTURE_GRAPH) {
      const auto &range_pair = capture_kernel_range_positions_.at(executors_[i].second);
      executor_mode_info << "executor range:[" << std::to_string(range_pair.first) << ", "
                         << std::to_string(range_pair.second) << "].";
    } else {
      executor_mode_info << "executor order:[" << std::to_string(executors_[i].second) << "]";
    }
    LOG_OUT << "The executor[" << i << "] is " << executor_mode << ", " << executor_mode_info.str();
  }

  return capture_graph_num_ > 0;
}

void GraphCaptureManager::CreateCaptureGraph(const device::DeviceContext *device_context, bool fullGraphMode) {
  LOG_OUT << "fullGraphMode: " << fullGraphMode;
  if (fullGraphMode) {
    LOG_OUT << "Cur shape: " << shape_key_;
    auto capture_graph = device_context->deviceResManager_->CreateCaptureGraph();
    CHECK_IF_NULL(capture_graph);
    capture_graphs_[shape_key_].emplace_back(capture_graph);
    return;
  }

  std::vector<CaptureGraphPtr> cur_capture_graphs;
  for (size_t i = 0; i < capture_graph_num_; i++) {
    // Create a capture graph using the device context
    auto capture_graph = device_context->deviceResManager_->CreateCaptureGraph();
    CHECK_IF_NULL(capture_graph);
    cur_capture_graphs.push_back(capture_graph);
  }
  capture_graphs_[shape_key_] = std::move(cur_capture_graphs);
}

bool GraphCaptureManager::LaunchAllKernelsWithCapture(std::vector<OpRunner> &opRunners) { return true; }

CaptureGraphPtr GraphCaptureManager::GetCurrentFullGraph() const {
  auto capture_graph_it = capture_graphs_.find(shape_key_);
  if (capture_graph_it == capture_graphs_.end() || capture_graph_it->second.empty()) {
    LOG_ERROR << "No capture graphs initialized for shape key: " << shape_key_;
    return nullptr;
  }

  CHECK_IF_FAIL(capture_graph_it->second.size() == 1);
  auto cur_capture_graph = capture_graph_it->second[0];
  CHECK_IF_NULL(cur_capture_graph);
  return cur_capture_graph;
}

void GraphCaptureManager::ExecuteCaptureOpsNeedNotUpdate(std::vector<OpRunner> &opRunners, size_t start, size_t end,
                                                         void *captureStream) const {
  for (size_t j = start; j <= end; j++) {
    auto &opRunner = opRunners[j];
    opRunner.UpdateTensors();
    if (auto errNo = opRunner.InferShape() != ops::SUCCESS) {
      LOG_EXCEPTION << "Infer shape failed for operator " << opRunner.GetOpName() << "Errno: " << errNo;
    }
    opRunner.AllocateMemory();
    if (auto errNo = opRunner.CalcWorkspace() != ops::SUCCESS) {
      LOG_EXCEPTION << "CalcWorkspace shape failed for operator " << opRunner.GetOpName() << "Errno: " << errNo;
    }
    opRunner.AllocateWorkspaceMemory();
    if (auto errNo = opRunner.Launch(captureStream) != ops::SUCCESS) {
      LOG_EXCEPTION << "Launch shape failed for operator " << opRunner.GetOpName() << "Errno: " << errNo;
    }
    if (IsCustomCallOp(opRunner)) {
      WaitLaunchTaskFinish();
    }
    opRunner.FreeMemory();
  }
}

void GraphCaptureManager::ExecuteCaptureOpNeedUpdate(OpRunner &opRunner, void *captureStream,
                                                     CaptureGraph *capture_graph, size_t single_op_index) {
  CHECK_IF_NULL(capture_graph);
  CHECK_IF_FAIL(single_op_index < singleOpPos_.size());

  auto &waitEvent = singleOpUpdateEvents_[single_op_index];
  CHECK_IF_NULL(waitEvent);

  opRunner.UpdateTensors();
  LOG_OUT << "Begin launch single op: " << opRunner.GetOpName()
          << ", executor index: " << singleOpPos_[single_op_index];
  if (auto errNo = opRunner.InferShape() != ops::SUCCESS) {
    LOG_EXCEPTION << "Infer shape failed for operator " << opRunner.GetOpName() << "Errno: " << errNo;
  }

  waitEvent->set_wait_stream(captureStream);
  waitEvent->WaitEventWithoutReset();
  waitEvent->ResetEvent();

  capture_graph->CaptureTaskGrpBegin(captureStream);
  opRunner.AllocateMemory();
  if (auto errNo = opRunner.CalcWorkspace() != ops::SUCCESS) {
    LOG_EXCEPTION << "CalcWorkspace shape failed for operator " << opRunner.GetOpName() << "Errno: " << errNo;
  }
  opRunner.AllocateWorkspaceMemory();
  if (auto errNo = opRunner.Launch(captureStream) != ops::SUCCESS) {
    LOG_EXCEPTION << "Launch shape failed for operator " << opRunner.GetOpName() << "Errno: " << errNo;
  }
  if (IsCustomCallOp(opRunner)) {
    // Custom call ops don't use InferRT updateStream_ to launch in update phase.
    LOG_EXCEPTION << "Not support update custom op in AclGraph fullgraph mode, got a costom op: "
                  << opRunner.GetOpName();
  }
  opRunner.FreeMemory();

  capture_graph->CaptureTaskGrpEnd(captureStream, &(singleOpUpdateHandles_[single_op_index]));
}

bool GraphCaptureManager::LaunchAllKernelsWithCaptureFullGraph(std::vector<OpRunner> &opRunners, void *captureStream) {
  LOG_OUT << "Begin launch all kernels with full graph capture, shape key: " << shape_key_;

  auto cur_capture_graph = GetCurrentFullGraph();
  CHECK_IF_NULL(cur_capture_graph);

  if (!cur_capture_graph->CaptureBegin(captureStream)) {
    LOG_ERROR << "Capture graph failed";
    return false;
  }

  LOG_OUT << "Begin full graph capture for shape key: " << shape_key_;
  LOG_OUT << "Begin launch all kernels with capture graph.";

  size_t singleOpCnt = 0;
  for (size_t i = 0; i < executors_.size(); i++) {
    auto &executor = executors_[i];
    if (executor.first == CAPTURE_GRAPH) {
      size_t start = capture_kernel_range_positions_[executor.second].first;
      size_t end = capture_kernel_range_positions_[executor.second].second;
      LOG_OUT << "Begin captrue graph, executor index: " << i << ", range[" << start << ", " << end << "].";
      ExecuteCaptureOpsNeedNotUpdate(opRunners, start, end, captureStream);
      continue;
    }

    auto &opRunner = opRunners[executor.second];
    ExecuteCaptureOpNeedUpdate(opRunner, captureStream, cur_capture_graph.get(), singleOpCnt);
    singleOpCnt++;
  }
  LOG_OUT << "End launch all kernels with capture graph.";

  cur_capture_graph->CaptureEnd(captureStream);
  LOG_OUT << "End full graph capture for shape key: " << shape_key_;
  LOG_OUT << "End launch all kernels with full graph capture.";
  return true;
}

bool GraphCaptureManager::LaunchAllKernelsWithReplayFullGraph(std::vector<OpRunner> &opRunners, void *executeStream,
                                                              void *updateStream) {
  LOG_OUT << "Begin launch all kernels with replay graph.";
  CHECK_IF_FAIL(capture_graphs_[shape_key_].size() == 1);
  auto &cur_capture_graph = capture_graphs_[shape_key_][0];  // Use first capture graph for full graph mode
  CHECK_IF_NULL(cur_capture_graph);
  cur_capture_graph->ExecuteCaptureGraph(executeStream);

  size_t singleOpNum = singleOpPos_.size();
  for (size_t i = 0; i < singleOpNum; ++i) {
    cur_capture_graph->CaptureTaskUpdateBegin(updateStream, singleOpUpdateHandles_[i]);

    auto &opRunner = opRunners[singleOpPos_[i]];
    if (auto errNo = opRunner.CalcWorkspace() != ops::SUCCESS) {
      LOG_EXCEPTION << "CalcWorkspace shape failed for operator " << opRunner.GetOpName() << "Errno: " << errNo;
    }
    opRunner.AllocateWorkspaceMemory();
    if (auto errNo = opRunner.Launch(updateStream) != ops::SUCCESS) {
      LOG_EXCEPTION << "Launch shape failed for operator " << opRunner.GetOpName() << "Errno: " << errNo;
    }
    if (IsCustomCallOp(opRunner)) {
      // Custom call ops don't use InferRT updateStream_ to launch in update phase, and Python call op maybe need copy
      // output from torch tensor to inferrt tensor, the src tensor ptr has been captured in aclgraph, use new allocated
      // output torch tenor memory is not correct.
      LOG_EXCEPTION << "Not support update custom op in AclGraph fullgraph mode, got a costom op: "
                    << opRunner.GetOpName();
    }
    opRunner.FreeWorkspaceMemory();

    cur_capture_graph->CaptureTaskUpdateEnd(updateStream);
    auto &waitEvent = singleOpUpdateEvents_[i];
    CHECK_IF_NULL(waitEvent);
    waitEvent->set_record_stream(updateStream);
    waitEvent->RecordEvent();
  }

  LOG_OUT << "End launch all kernels with replay graph.";
  return true;
}

void GraphCaptureManager::RecordGraphOutputKernelInfo(const OpRunner &opRunner, size_t index) {
  LOG_OUT << "Record current kernel: " << opRunner.GetOpName();

  // This is a simplified implementation - in a real scenario, we would extract
  // output tensors and record their device pointers, sizes, and shapes
  // For now, we'll just log the operation
}

void GraphCaptureManager::RecoverGraphOutputKernelInfo() {
  // This would restore output tensor information for graph replay
  // Implementation would depend on what was recorded in RecordGraphOutputKernelInfo
}

void GraphCaptureManager::UpdateFixAddressBeforeReplayGraph() {
  // This would update fixed addresses before replay
  // Implementation would update input tensors to their correct addresses
}

bool GraphCaptureManager::CheckParameterNotChange() {
  // This would check if parameters have changed between captures
  // For now, we'll return true to indicate no changes
  return true;
}

bool GraphCaptureManager::IsSingleOp(const std::vector<OpRunner> &opRunners, size_t kernel_index) {
  size_t executor_num = executors_.size();
  for (size_t i = 0; i < executor_num; i++) {
    auto &executor = executors_[i];
    if (executor.first != CAPTURE_GRAPH && executor.second == kernel_index) {
      return true;
    }
  }
  return false;
}

void GraphCaptureManager::InitFixedInputInfoForSingleOp(const std::vector<OpRunner> &opRunners) {
  size_t executor_num = executors_.size();
  for (size_t i = 0; i < executor_num; i++) {
    auto &executor = executors_[i];
    if (executor.first != CAPTURE_GRAPH) {
      auto kernel_idx = executor.second;
      auto &op_runner = opRunners[kernel_idx];
      // Use uintptr_t instead of pointer for hash key
      auto kernel_with_idx = std::make_pair(reinterpret_cast<uintptr_t>(&op_runner), kernel_idx);
      // Initialize fixed input info for single op replay
      std::vector<ir::ValuePtr> fix_input_kernel_tensors_for_single_op;
      // This would be populated with actual input tensors
      fix_network_input_for_replay_single_op_[shape_key_][kernel_with_idx] = fix_input_kernel_tensors_for_single_op;
    }
  }
}

bool GraphCaptureManager::IsNonFixedInputInReplay(const OpRunner &opRunner, size_t kernel_input_index) {
  // Check if this input is a non-fixed input (like weights or KV cache)
  // For now, we'll return false as a placeholder
  return false;
}

bool GraphCaptureManager::HasCapturedGraph() const {
  auto it = capture_graphs_.find(shape_key_);
  if (it != capture_graphs_.end()) {
    return true;
  }
  return false;
}

void GraphCaptureManager::Finalize() {
  capture_graphs_.clear();
  fixed_addrs_for_update_.clear();
  fixed_addrs_for_set_inputs_.clear();
  weight_kv_addrs_.clear();
  fix_single_op_input_info_.clear();
  fix_single_op_output_info_.clear();
  fix_single_op_workspace_info_.clear();
  fix_replay_graph_output_info_.clear();
  fix_network_input_for_replay_single_op_.clear();
  recorded_kernel_output_for_graph_output_.clear();
  task_groups_.clear();
  update_executors_.clear();

  // Clear other member variables
  capture_kernel_range_positions_.clear();
  executors_.clear();
  capture_graph_num_ = 0;
  init_ = false;
  shape_key_.clear();
}

}  // namespace runtime
}  // namespace mrt
