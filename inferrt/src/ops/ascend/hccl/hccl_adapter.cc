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
#include "ops/ascend/hccl/hccl_adapter.h"
#include <dlfcn.h>
#include <map>
#include <algorithm>
#include <sstream>

#include "hccl/hccl.h"
#include "hccl/hcom.h"

static constexpr const auto kHcclPluginFileName = "libhccl.so";

#define CHECK_SYMBOL_NULL(symbol)                                              \
  if ((symbol) == nullptr) {                                                   \
    LOG_ERROR << #symbol << " is null, hccl has not been inited, do nothing."; \
    return HcclResult::HCCL_E_RESERVED;                                        \
  }

namespace mrt::ops {
const char kDefaultGroup[] = "__default_group";
constexpr uint32_t kDeviceNumOfServer = 8;

HcclAdapter &HcclAdapter::GetInstance() {
  static HcclAdapter instance;
  return instance;
}

void HcclAdapter::InitPlugin() {
  if (plugin_handle_ != nullptr) {
    return;
  }
#ifndef ENABLE_ASAN
  plugin_handle_ = dlopen(kHcclPluginFileName, RTLD_DEEPBIND | RTLD_NOW | RTLD_LOCAL);
#else
  plugin_handle_ = dlopen(kHcclPluginFileName, RTLD_NOW | RTLD_LOCAL);
#endif
  if (plugin_handle_ == nullptr) {
    LOG_EXCEPTION << "Dlopen " << kHcclPluginFileName << " failed, result = " << GetDlErrorMsg();
  }

  launch_hccl_all_gather_ = DlsymFuncObj(HcclAllGather, plugin_handle_);
}

void HcclAdapter::FinalizePlugin() {
  if (plugin_handle_ == nullptr) {
    return;
  }
  set_hccl_global_comm_info_ = nullptr;
  init_hccl_root_info_config_ = nullptr;
  init_hccl_global_comm_ranktable_ = nullptr;
  init_hccl_sub_comm_ranktable_ = nullptr;
  get_hccl_comm_config_capability_ = nullptr;
  init_hccl_comm_ = nullptr;
  finalize_hccl_comm_ = nullptr;
  launch_hccl_broadcast_ = nullptr;
  launch_hccl_all_reduce_ = nullptr;
  launch_hccl_reduce_ = nullptr;
  launch_hccl_scatter_ = nullptr;
  launch_hccl_reduce_scatter_ = nullptr;
  launch_hccl_all_gather_ = nullptr;
  launch_hccl_send_ = nullptr;
  launch_hccl_recv_ = nullptr;
  launch_hccl_barrier_ = nullptr;
  launch_hccl_batch_isend_irecv_ = nullptr;
  hccl_create_group_ = nullptr;
  hccl_destroy_group_ = nullptr;
  hccl_get_rank_id_ = nullptr;
  hccl_get_local_rank_id_ = nullptr;
  hccl_get_local_rank_size_ = nullptr;
  hccl_get_world_rank_by_group_rank_ = nullptr;
  hccl_get_group_rank_by_world_rank_ = nullptr;
  hccl_get_rank_size_ = nullptr;
  hccl_exec_enqueue_op_ = nullptr;
  hccl_exec_enqueue_all_to_all_v_ = nullptr;
  hccl_comm_working_dev_nic_set_ = nullptr;
  launch_hccl_all_to_allv_ = nullptr;
  launch_hccl_reduce_scatterv_ = nullptr;
  launch_hccl_all_gatherv_ = nullptr;
  launch_hccl_comm_resume_ = nullptr;
  hcom_destroy_ = nullptr;
  (void)dlclose(plugin_handle_);
  plugin_handle_ = nullptr;
}

std::string HcclAdapter::GetHcclModeString(HcclMode hccl_mode) {
  static std::map<HcclMode, std::string> kHcclModeString = {{HcclMode::kGraph, "GE_MODE"},
                                                            {HcclMode::kPynative, "PYNATIVE_MODE"},
                                                            {HcclMode::kKernelByKernel, "KERNEL_BY_KERNEL_MODE"}};
  return kHcclModeString.at(hccl_mode);
}

bool HcclAdapter::InitHccl(uint32_t device_id, std::string_view rank_id) {
  LOG_OUT << "Start init hccl adapter.";
  std::lock_guard<std::mutex> lock(init_mutex_);
  if (init_flag_) {
    LOG_OUT << "Hccl has been inited, skip.";
    return true;
  }
  InitPlugin();

  init_flag_ = true;
  LOG_OUT << "Init hccl adapter success.";
  return true;
}

bool HcclAdapter::HcclWatchdogThread(HcclComm comm, std::string *error_info, bool *disable) {
  if (!init_flag_) {
    LOG_OUT << "Hccl has never been inited, skip.";
    return true;
  }
  CHECK_IF_NULL(disable);
  if (hccl_get_comm_async_error_ == nullptr) {
    LOG_OUT << "Hccl has never been inited, skip.";
    return true;
  }
  if (hccl_get_error_string_ == nullptr) {
    LOG_OUT << "Hccl has never been inited, skip.";
    return true;
  }
  HcclResult hccl_async_error;
  auto ret = hccl_get_comm_async_error_(comm, &hccl_async_error);
  if (ret != HCCL_SUCCESS) {
    LOG_OUT << "Call HcclGetCommAsyncError failed, close watchdog.";
    *disable = true;
    return true;
  }
  if (hccl_async_error != HCCL_SUCCESS) {
    std::ostringstream oss;
    oss << "Hccl get comm async error failed, error code is: " << hccl_async_error
        << ", detail info: " << hccl_get_error_string_(hccl_async_error);
    *error_info = oss.str();
    return false;
  }
  return true;
}

bool HcclAdapter::FinalizeHccl() {
  std::lock_guard<std::mutex> lock(init_mutex_);
  LOG_OUT << "Start destroy hccl adapter for " << GetHcclModeString(hccl_mode_);
  if (!init_flag_) {
    LOG_OUT << "Hccl has never been inited, skip.";
    return true;
  }
  (void)FinalizeHcclExec();
  (void)FinalizeKernelInfoStore();
  (void)FinalizeHcclComm();
  if (hcom_destroy_ != nullptr) {
    hcom_destroy_();
  }
  FinalizePlugin();
  init_flag_ = false;
  LOG_OUT << "Destroy hccl adapter success.";
  return true;
}

HcclResult HcclAdapter::HcclBroadcast(void *buf, uint64_t count, HcclDataType dataType, uint32_t root,
                                      aclrtStream stream, HcclComm hccl_comm) const {
  HcclResult ret = launch_hccl_broadcast_(buf, count, dataType, root, hccl_comm, stream);

  return ret;
}

HcclResult HcclAdapter::HcclAllReduce(void *send_buf, void *recv_buf, uint64_t count, HcclDataType dataType,
                                      const HcclReduceOp op, const aclrtStream stream, HcclComm hccl_comm) const {
  HcclResult ret = launch_hccl_all_reduce_(send_buf, recv_buf, count, dataType, op, hccl_comm, stream);

  return ret;
}

HcclResult HcclAdapter::HcclReduce(void *send_buf, void *recv_buf, uint64_t count, HcclDataType dataType,
                                   HcclReduceOp op, uint32_t root, const aclrtStream stream, HcclComm hccl_comm) const {
  HcclResult ret = launch_hccl_reduce_(send_buf, recv_buf, count, dataType, op, root, hccl_comm, stream);

  return ret;
}

HcclResult HcclAdapter::HcclScatter(void *send_buf, void *recv_buf, uint64_t count, HcclDataType dataType,
                                    uint32_t root, HcclComm comm, aclrtStream stream) const {
  HcclResult ret = launch_hccl_scatter_(send_buf, recv_buf, count, dataType, root, comm, stream);

  return ret;
}

HcclResult HcclAdapter::HcclReduceScatter(void *send_buf, void *recv_buf, uint64_t count, HcclDataType dataType,
                                          const HcclReduceOp op, const aclrtStream stream, HcclComm hccl_comm) const {
  HcclResult ret = launch_hccl_reduce_scatter_(send_buf, recv_buf, count, dataType, op, hccl_comm, stream);

  return ret;
}

HcclResult HcclAdapter::HcclAllGather(void *send_buf, void *recv_buf, uint64_t count, HcclDataType dataType,
                                      const aclrtStream stream, HcclComm hccl_comm) const {
  CHECK_SYMBOL_NULL(launch_hccl_all_gather_);
  CHECK_IF_NULL(hccl_comm);
  CHECK_IF_NULL(send_buf);
  CHECK_IF_NULL(recv_buf);
  HcclResult ret = launch_hccl_all_gather_(send_buf, recv_buf, count, dataType, hccl_comm, stream);
  return ret;
}

HcclResult HcclAdapter::HcclSend(void *send_buf, uint64_t count, HcclDataType dataType, uint32_t destRank,
                                 const aclrtStream stream, HcclComm hccl_comm) const {
  HcclResult ret = launch_hccl_send_(send_buf, count, dataType, destRank, hccl_comm, stream);

  return ret;
}

HcclResult HcclAdapter::HcclRecv(void *recv_buf, uint64_t count, HcclDataType dataType, uint32_t srcRank,
                                 const aclrtStream stream, HcclComm hccl_comm) const {
  HcclResult ret = launch_hccl_recv_(recv_buf, count, dataType, srcRank, hccl_comm, stream);

  return ret;
}

HcclResult HcclAdapter::HcclBarrier(const aclrtStream stream, HcclComm hccl_comm) const {
  return launch_hccl_barrier_(hccl_comm, stream);
}

HcclResult HcclAdapter::HcclBatchISendIRecv(HcclSendRecvItem *sendRecvInfo, uint32_t itemNum, HcclComm comm,
                                            aclrtStream stream) const {
  HcclResult ret = launch_hccl_batch_isend_irecv_(sendRecvInfo, itemNum, comm, stream);

  return ret;
}

HcclResult HcclAdapter::HcclCommResume(HcclComm comm) const {
  if (launch_hccl_comm_resume_ == nullptr) {
    LOG_EXCEPTION << "Dynamically load HcclCommResume failed.";
  }
  return launch_hccl_comm_resume_(comm);
}

uint32_t HcclAdapter::HcclGetCommConfigCapability() {
  CHECK_IF_NULL(get_hccl_comm_config_capability_);
  return get_hccl_comm_config_capability_();
}

HcclResult HcclAdapter::HcclSetGlobalCommInfo(uint32_t masterIp, uint32_t masterPort, uint32_t totalRankSize,
                                              uint32_t nodeId, uint32_t localRankSize) {
  if (set_hccl_global_comm_info_ == nullptr) {
    set_hccl_global_comm_info_ = DlsymAscendFuncObj(HcclSetGlobalCommInfo, plugin_handle_);
    if (set_hccl_global_comm_info_ == nullptr) {
      LOG_OUT << "Func HcclSetGlobalCommInfo is not supported in CANN package.";
      return HCCL_E_NOT_SUPPORT;
    }
  }
  return set_hccl_global_comm_info_(masterIp, masterPort, totalRankSize, nodeId, localRankSize);
}

HcclResult HcclAdapter::HcclCommInitClusterInfoConfig(const char *rank_table, uint32_t rank_id, HcclCommConfig *config,
                                                      HcclComm *hccl_comm) {
  if (init_hccl_global_comm_ranktable_ == nullptr) {
    init_hccl_global_comm_ranktable_ = DlsymFuncObj(HcclCommInitClusterInfoConfig, plugin_handle_);
  }
  return init_hccl_global_comm_ranktable_(rank_table, rank_id, config, hccl_comm);
}

HcclResult HcclAdapter::HcclCommInitRootInfoConfig(uint32_t n_ranks, const HcclRootInfo *root_info, uint32_t rank,
                                                   const HcclCommConfig *config, HcclComm *hccl_comm_) {
  if (init_hccl_root_info_config_ == nullptr) {
    init_hccl_root_info_config_ = DlsymFuncObj(HcclCommInitRootInfoConfig, plugin_handle_);
    if (init_hccl_root_info_config_ == nullptr) {
      // new api in CANN C20
      return HcclCommInitRootInfo(n_ranks, root_info, rank, hccl_comm_);
    }
  }

  return init_hccl_root_info_config_(n_ranks, root_info, rank, config, hccl_comm_);
}

HcclResult HcclAdapter::HcclCreateSubCommConfig(HcclComm *global_comm, uint32_t rank_size, uint32_t *rank_ids,
                                                uint64_t comm_id, uint32_t rank_id, HcclCommConfig *config,
                                                HcclComm *hccl_comm) {
  if (init_hccl_sub_comm_ranktable_ == nullptr) {
    init_hccl_sub_comm_ranktable_ = DlsymFuncObj(HcclCreateSubCommConfig, plugin_handle_);
  }
  return init_hccl_sub_comm_ranktable_(global_comm, rank_size, rank_ids, comm_id, rank_id, config, hccl_comm);
}

bool HcclAdapter::InitHcclComm(std::string_view rank_id, std::string_view rank_file) {
  LOG_OUT << "Start init hccl comm.";
  int rank_id_i = -1;
  try {
    rank_id_i = std::stoi(rank_id.data());
  } catch (std::invalid_argument &) {
    LOG_EXCEPTION << "Invalid rank id env:" << rank_id;
  }
  if (rank_id_i < 0) {
    LOG_ERROR << "rank_id cannot be negative";
    return false;
  }
  CHECK_IF_NULL(init_hccl_comm_);
  auto hccl_result = init_hccl_comm_(rank_file.data(), rank_id_i, &hccl_comm_);
  if (hccl_result != HCCL_SUCCESS) {
    LOG_ERROR << "HcclCommInitClusterInfo failed, ret:" << hccl_result;
    return false;
  }
  LOG_OUT << "InitHcclComm success";
  return true;
}

bool HcclAdapter::FinalizeHcclComm() {
  LOG_OUT << "Start finalize hccl comm.";
  if (hccl_comm_ == nullptr) {
    return true;
  }

  CHECK_IF_NULL(finalize_hccl_comm_);
  auto hccl_result = finalize_hccl_comm_(hccl_comm_);
  if (hccl_result != HCCL_SUCCESS) {
    LOG_ERROR << "HcclComm destroy failed, ret:" << hccl_result;
    return false;
  }
  hccl_comm_ = nullptr;
  LOG_OUT << "HcclComm destroy success";
  return true;
}

HcclResult HcclAdapter::HcclCreateGroup(const std::string &group, uint32_t rank_num, uint32_t *rank_ids) const {
  CHECK_SYMBOL_NULL(hccl_create_group_);
  return hccl_create_group_(group.c_str(), rank_num, rank_ids);
}

HcclResult HcclAdapter::HcclDestroyGroup(const std::string &group) const {
  CHECK_SYMBOL_NULL(hccl_destroy_group_);
  return hccl_destroy_group_(group.c_str());
}

HcclResult HcclAdapter::HcclGetRankId(const std::string &group, uint32_t *rank_id) const {
  if (hccl_mode_ != HcclMode::kGraph) {
    CHECK_SYMBOL_NULL(single_op_hccl_get_rank_id_);
    return single_op_hccl_get_rank_id_(hccl_comm_, rank_id);
  } else {
    CHECK_SYMBOL_NULL(hccl_get_rank_id_);
    return hccl_get_rank_id_(group.c_str(), rank_id);
  }
}

HcclResult HcclAdapter::HcclGetRankSize(const std::string &group, uint32_t *rank_size) const {
  if (hccl_mode_ != HcclMode::kGraph) {
    CHECK_SYMBOL_NULL(single_op_hccl_get_rank_size_);
    return single_op_hccl_get_rank_size_(hccl_comm_, rank_size);
  } else {
    CHECK_SYMBOL_NULL(hccl_get_rank_size_);
    return hccl_get_rank_size_(group.c_str(), rank_size);
  }
}

HcclResult HcclAdapter::HcclGetLocalRankId(const std::string &group, uint32_t *local_rank_id) const {
  CHECK_SYMBOL_NULL(hccl_get_local_rank_id_);
  return hccl_get_local_rank_id_(group.c_str(), local_rank_id);
}

HcclResult HcclAdapter::HcclGetLocalRankSize(const std::string &group, uint32_t *local_rank_size) const {
  if (hccl_mode_ != HcclMode::kGraph) {
    LOG_ERROR << "The pynative mode doesn't support get local rank szie.";
    return HCCL_E_NOT_SUPPORT;
  } else {
    CHECK_SYMBOL_NULL(hccl_get_local_rank_size_);
    return hccl_get_local_rank_size_(group.c_str(), local_rank_size);
  }
}

HcclResult HcclAdapter::HcclGetWorldRankFromGroupRank(const std::string &group, uint32_t local_rank,
                                                      uint32_t *world_rank) const {
  if (hccl_mode_ != HcclMode::kGraph) {
    LOG_ERROR << "The pynative mode doesn't support get world rank by group rank.";
    return HCCL_E_NOT_SUPPORT;
  } else {
    CHECK_SYMBOL_NULL(hccl_get_world_rank_by_group_rank_);
    return hccl_get_world_rank_by_group_rank_(group.c_str(), local_rank, world_rank);
  }
}

HcclResult HcclAdapter::HcclGetGroupRankFromWorldRank(uint32_t world_rank, const std::string &group,
                                                      uint32_t *local_rank) const {
  if (hccl_mode_ != HcclMode::kGraph) {
    LOG_ERROR << "The pynative mode doesn't support get group rank by world rank.";
    return HCCL_E_NOT_SUPPORT;
  } else {
    CHECK_SYMBOL_NULL(hccl_get_group_rank_by_world_rank_);
    return hccl_get_group_rank_by_world_rank_(world_rank, group.c_str(), local_rank);
  }
}

HcclResult HcclAdapter::HcclCommWorkingDevNicSet(HcclComm comm, uint32_t *ranks, bool *useBackup, uint32_t nRanks) {
  if (hccl_comm_working_dev_nic_set_ == nullptr) {
    hccl_comm_working_dev_nic_set_ = DlsymFuncObj(HcclCommWorkingDevNicSet, plugin_handle_);
  }
  CHECK_SYMBOL_NULL(hccl_comm_working_dev_nic_set_);
  return hccl_comm_working_dev_nic_set_(comm, ranks, useBackup, nRanks);
}

HcclResult HcclAdapter::HcclExecEnqueueOp(const ::HcomOperation &op_info, const HExecCallBack &callback) const {
  CHECK_SYMBOL_NULL(hccl_exec_enqueue_op_);
  return hccl_exec_enqueue_op_(op_info, callback);
}

HcclResult HcclAdapter::HcclExecAlltoAllV(const ::HcomAllToAllVParams &params, const HExecCallBack &callback) const {
  CHECK_SYMBOL_NULL(hccl_exec_enqueue_all_to_all_v_);
  return hccl_exec_enqueue_all_to_all_v_(params, callback);
}

bool HcclAdapter::UseHcclCM() const {
  // This MS_HCCL_CM_INIT env is deperacated since MindSpore 2.3 version.
  return false;
}

HcclResult HcclAdapter::HcclAlltoAllV(void *send_buf, void *recv_buf, HcclAllToAllVParams params, HcclDataType dataType,
                                      aclrtStream stream, HcclComm hccl_comm) const {
  CHECK_SYMBOL_NULL(launch_hccl_all_to_allv_);
  CHECK_IF_NULL(hccl_comm);
  HcclResult ret =
    launch_hccl_all_to_allv_(send_buf, params.sendcounts.data(), params.sdispls.data(), dataType, recv_buf,
                             params.recvcounts.data(), params.rdispls.data(), dataType, hccl_comm, stream);

  return ret;
}

HcclResult HcclAdapter::HcclReduceScatterV(void *send_buf, void *recv_buf, HcclReduceScatterVParams params,
                                           HcclDataType data_type, const HcclReduceOp op, const aclrtStream stream,
                                           HcclComm hccl_comm) const {
  CHECK_SYMBOL_NULL(launch_hccl_reduce_scatterv_);
  CHECK_IF_NULL(hccl_comm);
  HcclResult ret = launch_hccl_reduce_scatterv_(send_buf, params.send_counts.data(), params.sdispls.data(), recv_buf,
                                                params.recv_count, data_type, op, hccl_comm, stream);
  return ret;
}

HcclResult HcclAdapter::HcclAllGatherV(void *send_buf, void *recv_buf, HcclAllGatherVParams params,
                                       HcclDataType data_type, const aclrtStream stream, HcclComm hccl_comm) const {
  CHECK_SYMBOL_NULL(launch_hccl_all_gatherv_);
  CHECK_IF_NULL(hccl_comm);
  HcclResult ret = launch_hccl_all_gatherv_(send_buf, params.send_count, recv_buf, params.recv_counts.data(),
                                            params.rdispls.data(), data_type, hccl_comm, stream);
  return ret;
}

HcclResult HcclAdapter::HcclAllToAll(void *send_buf, void *recv_buf, HcclAllToAllParams params, HcclDataType dataType,
                                     aclrtStream stream, HcclComm hccl_comm) const {
  CHECK_SYMBOL_NULL(launch_hccl_all_to_all_);
  CHECK_IF_NULL(hccl_comm);

  HcclResult ret = launch_hccl_all_to_all_(send_buf, params.sendcount, dataType, recv_buf, params.recvcount, dataType,
                                           hccl_comm, stream);

  return ret;
}

bool HcclAdapter::IsSameServer(const std::vector<uint32_t> &rank_ids) const {
  auto min_iter = min_element(rank_ids.begin(), rank_ids.end());
  uint32_t min = (min_iter != rank_ids.end()) ? *min_iter : 0;
  auto max_iter = max_element(rank_ids.begin(), rank_ids.end());
  uint32_t max = (max_iter != rank_ids.end()) ? *max_iter : 0;
  return ((max - min < kDeviceNumOfServer) && (min / kDeviceNumOfServer == max / kDeviceNumOfServer));
}

}  // namespace mrt::ops
