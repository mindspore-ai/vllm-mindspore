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

#include "hardware/ascend/res_manager/ascend_stream_manager.h"

#include <string>
#include "common/common.h"
#include "hardware/hardware_abstract/common.h"
#include "acl/error_codes/rt_error_codes.h"
#include "hardware/ascend/res_manager/mem_manager/ascend_gmem_adapter.h"
#include "hardware/ascend/res_manager/symbol_interface/acl_rt_symbol.h"
#include "hardware/ascend/res_manager/symbol_interface/symbol_utils.h"

namespace mrt {
namespace device {
namespace ascend {
namespace {
constexpr size_t kIndex0 = 0;
}
AscendStreamMng &AscendStreamMng::GetInstance() {
  static AscendStreamMng instance{};
  return instance;
}

void AscendStreamMng::DestroyAllRtEvents() {
  for (size_t i = 0; i < events_.size(); ++i) {
    if (events_[i] != nullptr) {
      auto rt_ret = CALL_ASCEND_API(aclrtDestroyEvent, events_[i]);
      if (rt_ret != ACL_SUCCESS) {
        LOG_ERROR << "Call aclrtDestroyEvent failed, ret:" << rt_ret;
      }
    }
  }
  events_.clear();
}

void AscendStreamMng::DeleteEvent() {
  if (curEventNum_ == 0) {
    LOG_OUT << "total event num is 0, no event to delete";
  } else {
    --curEventNum_;
  }
}

void AscendStreamMng::DeleteStream() {
  if (curStreamNum_ == 0) {
    LOG_OUT << " total stream num is 0, no stream to delete";
  } else {
    --curStreamNum_;
  }
}

uint32_t AscendStreamMng::GetCurAllocStreamId() const {
  if (curStreamNum_ == 0) {
    LOG_ERROR << "stream nums is 0, no stream id should be get";
  }
  return curStreamNum_ - 1;
}

void AscendStreamMng::CreateStream(aclrtStream *stream, int32_t priority) {
  std::lock_guard<std::mutex> lock_streams(streamMutex_);
  auto ret = CALL_ASCEND_API(aclrtCreateStreamWithConfig, stream, IntToUint(priority),
                             (ACL_STREAM_FAST_LAUNCH | ACL_STREAM_FAST_SYNC));
  if (ret != ACL_SUCCESS) {
    LOG_ERROR << "Create stream failed, ret:" << ret;
  }
  ret = CALL_ASCEND_API(aclrtSetStreamFailureMode, *stream, ACL_STOP_ON_FAILURE);
  if (ret != ACL_SUCCESS) {
    LOG_ERROR << "aclrtSetStreamFailureMode failed, ret:" << ret;
  }
  (void)streams_.emplace_back(*stream);
}

void AscendStreamMng::CreateStream(size_t *stream_id, int32_t priority) {
  std::lock_guard<std::mutex> lock_streams(streamMutex_);
  aclrtStream stream;
  auto ret = CALL_ASCEND_API(aclrtCreateStreamWithConfig, &stream, IntToUint(priority),
                             (ACL_STREAM_FAST_LAUNCH | ACL_STREAM_FAST_SYNC));
  if (ret != ACL_SUCCESS) {
    LOG_ERROR << "Create stream failed, ret:" << ret;
  }
  ret = CALL_ASCEND_API(aclrtSetStreamFailureMode, stream, ACL_STOP_ON_FAILURE);
  if (ret != ACL_SUCCESS) {
    LOG_ERROR << "aclrtSetStreamFailureMode failed, ret:" << ret;
  }
  *stream_id = streams_.size();
  (void)streams_.emplace_back(stream);
}

void AscendStreamMng::CreateStreamWithFlags(aclrtStream *stream, uint32_t flags, int32_t priority) {
  std::lock_guard<std::mutex> lock_streams(streamMutex_);
  auto ret = CALL_ASCEND_API(aclrtCreateStreamWithConfig, stream, IntToUint(priority), flags);
  if (ret != ACL_SUCCESS) {
    LOG_ERROR << "Create stream failed, ret:" << ret;
  }
  ret = CALL_ASCEND_API(aclrtSetStreamFailureMode, *stream, ACL_STOP_ON_FAILURE);
  if (ret != ACL_SUCCESS) {
    LOG_ERROR << "aclrtSetStreamFailureMode failed, ret:" << ret;
  }
  (void)streams_.emplace_back(*stream);
}

void AscendStreamMng::CreateStreamWithFlags(size_t *stream_id, uint32_t flags, int32_t priority) {
  std::lock_guard<std::mutex> lock_streams(streamMutex_);
  aclrtStream stream;
  auto ret = CALL_ASCEND_API(aclrtCreateStreamWithConfig, &stream, IntToUint(priority), flags);
  if (ret != ACL_SUCCESS) {
    LOG_ERROR << "Create stream failed, ret:" << ret;
  }
  ret = CALL_ASCEND_API(aclrtSetStreamFailureMode, stream, ACL_STOP_ON_FAILURE);
  if (ret != ACL_SUCCESS) {
    LOG_ERROR << "aclrtSetStreamFailureMode failed, ret:" << ret;
  }
  *stream_id = streams_.size();
  (void)streams_.emplace_back(stream);
}

aclrtEvent AscendStreamMng::ApplyRtEvent() {
  aclrtEvent rt_event = nullptr;
  // Use ex api of event, so that no limits on event total size.
  uint32_t flag = ACL_EVENT_SYNC;
  auto ret = CALL_ASCEND_API(aclrtCreateEventExWithFlag, &rt_event, flag);
  if (ret != ACL_SUCCESS) {
    LOG_ERROR << "aclrtCreateEventExWithFlag failed, ret : " << ret << ".";
  }
  (void)events_.emplace_back(rt_event);
  return rt_event;
}

bool AscendStreamMng::DestroyStream(size_t stream_id) {
  std::lock_guard<std::mutex> lock_streams(streamMutex_);
  if (stream_id >= streams_.size()) {
    LOG_ERROR << "Ascend stream not found for stream id " << stream_id;
    return false;
  }
  if (streams_.at(stream_id) == nullptr) {
    LOG_OUT << "Ascend stream hsa been destroyed for stream id " << stream_id;
    return true;
  }
  const auto ret = CALL_ASCEND_API(aclrtDestroyStream, streams_.at(stream_id));
  if (ret != ACL_SUCCESS) {
    LOG_ERROR << "Call aclrtDestroyStream, ret[" << ret << "]";
  }
  streams_[stream_id] = nullptr;
  if (communicationStreamId_ == stream_id) {
    communicationStream_ = nullptr;
  }
  if (defaultStreamId_ == stream_id) {
    defaultStream_ = nullptr;
  }

  return true;
}

bool AscendStreamMng::ForceDestroyAllStreams() {
  std::lock_guard<std::mutex> lock_streams(streamMutex_);
  for (const auto &stream : streams_) {
    if (stream == nullptr) {
      continue;
    }
    const auto ret = CALL_ASCEND_API(aclrtDestroyStreamForce, stream);
    if (ret != ACL_SUCCESS) {
      LOG_ERROR << "Call aclrtDestroyStream, ret[" << ret << "]";
    }
  }
  streams_.clear();
  defaultStream_ = nullptr;
  communicationStream_ = nullptr;
  return true;
}

bool AscendStreamMng::DestroyAllStreams() {
  std::lock_guard<std::mutex> lock_streams(streamMutex_);
  for (const auto &stream : streams_) {
    if (stream == nullptr) {
      continue;
    }
    const auto ret = CALL_ASCEND_API(aclrtDestroyStream, stream);
    if (ret != ACL_SUCCESS) {
      LOG_ERROR << "Call aclrtDestroyStream, ret[" << ret << "]";
    }
  }
  streams_.clear();
  defaultStream_ = nullptr;
  communicationStream_ = nullptr;
  return true;
}

aclrtStream AscendStreamMng::GetStream(size_t stream_id) const {
  if (stream_id >= streams_.size()) {
    LOG_OUT << "Stream for stream id[" << stream_id << "] not found, return nullptr.";
    return nullptr;
  }
  return streams_[stream_id];
}

bool AscendStreamMng::SyncStream(size_t stream_id) const {
  if (stream_id >= streams_.size()) {
    LOG_ERROR << "Stream for stream id[" << stream_id << "] has not been created.";
  }
  const auto stream = streams_[stream_id];
  if (stream == nullptr) {
    LOG_OUT << "Stream for stream id[" << stream_id << "] has been destroyed.";
    return false;
  }
  return SyncStream(stream);
}

bool AscendStreamMng::SyncStream(aclrtStream stream) const {
  CHECK_IF_NULL(stream);
  LOG_OUT << "Sync stream: " << stream;
  auto RET = ACL_SUCCESS;
  try {
    GilReleaseWithCheck gil_release;
    RET = CALL_ASCEND_API(aclrtSynchronizeStreamWithTimeout, stream, -1);
    if (RET != ACL_SUCCESS && RET != ACL_ERROR_RT_AICORE_OVER_FLOW) {  // o for switch stream
      LOG_ERROR << "Call runtime aclrtSynchronizeStreamWithTimeout error."
                << "Please do the following three things to confirm whether it is caused by the "
                << "execution failure of a certain operator.\n"
                << "    1.Set inferrt.runtime.launch_blocking() at the beginning of your python script.\n"
                << "    2.Run again your python script.\n"
                << "    3.Grep 'Sync run failed' in your logs, it always stays at the end of your logs.\n"
                << "Now you will get the certain failed op detailed infos.";
      return false;
    }
  } catch (const std::exception &e) {
    LOG_ERROR << "Sync stream failed. " << e.what()
              << "Please do the following three things to confirm whether it is caused by the "
              << "execution failure of a certain operator.\n"
              << "    1.Set inferrt.runtime.launch_blocking() at the beginning of your python script.\n"
              << "    2.Run again your python script.\n"
              << "    3.Grep 'Sync run failed' in your logs, it always stays at the end of your logs.\n"
              << "Now you will get the certain failed op detailed infos.";
    return false;
  }
  if (RET == ACL_ERROR_RT_AICORE_OVER_FLOW) {
    LOG_OUT << "Call runtime aclrtSynchronizeStreamWithTimeout, the stream get overflow.";
  }
  return true;
}

bool AscendStreamMng::SyncAllStreams(bool sync_device) const {
  auto RET = ACL_ERROR_NONE;
  try {
    GilReleaseWithCheck gil_release;
    if (sync_device) {
      // According to CANN, we need to set timeout to 2 hours for aclrtSynchronizeDeviceWithTimeout.
      int timeout = 7200000;
      RET = CALL_ASCEND_API(aclrtSynchronizeDeviceWithTimeout, timeout);
      if (RET != ACL_ERROR_NONE && RET != ACL_ERROR_RT_AICORE_OVER_FLOW) {
        LOG_ERROR << "Call runtime aclrtSynchronizeDeviceWithTimeout error."
                  << "Please do the following three things to confirm whether it is caused by the "
                  << "execution failure of a certain operator.\n"
                  << "    1.Set inferrt.runtime.launch_blocking() at the beginning of your python script.\n"
                  << "    2.Run again your python script.\n"
                  << "    3.Grep 'Sync run failed' in your logs, it always stays at the end of your logs.\n"
                  << "Now you will get the certain failed op detailed infos.";
        return false;
      }
    } else {
      for (size_t i = 0; i < streams_.size(); i++) {
        const auto stream = streams_[i];
        if (stream != nullptr && !SyncStream(stream)) {
          LOG_ERROR << "SyncStream for stream id " << i << " failed.";
          return false;
        }
      }
    }
  } catch (const std::exception &e) {
    std::string sync_method = sync_device ? "aclrtSynchronizeDeviceWithTimeout" : "aclrtSynchronizeStreamWithTimeout";
    LOG_ERROR << sync_method << " failed. " << e.what()
              << "Please do the following three things to confirm whether it is caused by the "
              << "execution failure of a certain operator.\n"
              << "    1.Set inferrt.runtime.launch_blocking() at the beginning of your python script.\n"
              << "    2.Run again your python script.\n"
              << "    3.Grep 'Sync run failed' in your logs, it always stays at the end of your logs.\n"
              << "Now you will get the certain failed op detailed infos.";
    return false;
  }
  if (RET == ACL_ERROR_RT_AICORE_OVER_FLOW) {
    std::string sync_method = sync_device ? "aclrtSynchronizeDeviceWithTimeout" : "aclrtSynchronizeStreamWithTimeout";
    LOG_OUT << "Call runtime " << sync_method << ", the stream get overflow."
            << "Please do the following three things to confirm whether it is caused by the "
            << "execution failure of a certain operator.\n"
            << "    1.Set inferrt.runtime.launch_blocking() at the beginning of your python script.\n"
            << "    2.Run again your python script.\n"
            << "    3.Grep 'Sync run failed' in your logs, it always stays at the end of your logs.\n"
            << "Now you will get the certain failed op detailed infos.";
  }
  return true;
}

bool AscendStreamMng::SyncNotDefaultStreams() const {
  bool res = true;
  for (size_t i = 0; i < streams_.size(); i++) {
    if (i != defaultStreamId_ && !SyncStream(i)) {
      LOG_ERROR << "Failed to sync for ascend stream id: " << i;
      res = false;
    }
  }
  return res;
}

bool AscendStreamMng::SyncExceptStreamsInList(const std::set<aclrtStream> &except_streams) const {
  bool res = true;
  for (size_t i = 0; i < streams_.size(); i++) {
    if (except_streams.count(streams_[i]) > 0) {
      LOG_OUT << "Stream id:" << i << " is been synchronized.";
      continue;
    }
    if (!SyncStream(i)) {
      LOG_ERROR << "Failed to sync for ascend stream id: " << i;
      res = false;
    }
  }
  return res;
}

size_t AscendStreamMng::QueryStreamSize() const { return streams_.size(); }

bool AscendStreamMng::QueryStream(size_t stream_id) {
  if (stream_id >= streams_.size()) {
    LOG_ERROR << "Stream for stream id[" << stream_id << "] has not been created.";
  }
  const auto stream = streams_[stream_id];
  if (stream == nullptr) {
    LOG_OUT << "Stream for stream id[" << stream_id << "] has been destroyed.";
    return false;
  }

  aclrtStreamStatus status;
  auto ret = CALL_ASCEND_API(aclrtStreamQuery, stream, &status);
  if (ret != ACL_SUCCESS) {
    LOG_ERROR << "Failed to query completion status for stream id: " << stream_id;
  }
  return status == ACL_STREAM_STATUS_COMPLETE;
}

size_t AscendStreamMng::GetStreamId(void *stream_ptr) {
  auto iter = std::find(streams_.begin(), streams_.end(), stream_ptr);
  if (iter == streams_.end()) {
    LOG_ERROR << "Failed to find stream_ptr in streams_, stream_ptr:" << stream_ptr;
  }

  return LongToSize(std::distance(streams_.begin(), iter));
}

std::vector<uint32_t> AscendStreamMng::GetStreamIds() const {
  std::vector<uint32_t> stream_ids;
  for (size_t i = 0; i < streams_.size(); i++) {
    if (streams_[i] != nullptr) {
      (void)stream_ids.emplace_back(static_cast<uint32_t>(i));
    }
  }
  return stream_ids;
}

void AscendStreamMng::CreateDefaultStream() {
  if (defaultStream_ == nullptr) {
    CreateStream(&defaultStreamId_);
    LOG_OUT << "Create ascend default stream, stream id: " << defaultStreamId_;
    defaultStream_ = GetStream(defaultStreamId_);
    CHECK_IF_NULL(defaultStream_);
  } else {
    LOG_OUT << "The default compute stream is already created, skip.";
  }

  if (communicationStream_ == nullptr) {
    CreateStream(&communicationStreamId_);
    LOG_OUT << "Create ascend communication stream, stream id: " << communicationStreamId_;
    communicationStream_ = GetStream(communicationStreamId_);
    CHECK_IF_NULL(communicationStream_);
  } else {
    LOG_OUT << "The default communication stream is already created, skip.";
  }
}

size_t AscendStreamMng::default_stream_id() const {
  if (defaultStream_ == nullptr) {
    LOG_ERROR << "The default stream is not created";
  }
  return defaultStreamId_;
}
size_t AscendStreamMng::communication_stream_id() const {
  if (communicationStream_ == nullptr) {
    LOG_ERROR << "The communication stream is not created";
  }
  return communicationStreamId_;
}
aclrtStream AscendStreamMng::default_stream() const { return defaultStream_; }
aclrtStream AscendStreamMng::communication_stream() const { return communicationStream_; }

}  // namespace ascend
}  // namespace device
}  // namespace mrt
