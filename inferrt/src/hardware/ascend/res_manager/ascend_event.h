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

#ifndef INFERRT_SRC_HARDWARE_ASCEND_ASCEND_EVENT_H
#define INFERRT_SRC_HARDWARE_ASCEND_ASCEND_EVENT_H

#include "hardware/hardware_abstract/device_event.h"
#include "acl/acl_rt.h"
#include "common/visible.h"

namespace mrt::device::ascend {
constexpr uint32_t ACL_EVENT_DEFAULT = 0x0000000Eu;

class MRT_EXPORT AscendEvent : public DeviceEvent {
 public:
  AscendEvent();
  explicit AscendEvent(uint32_t flag, bool useExtensionalApi = true);
  ~AscendEvent() override;

  bool IsReady() const override;
  void WaitEvent() override;
  bool WaitEvent(uint32_t streamId) override;
  void WaitEventWithoutReset() override;
  void WaitEventWithoutReset(uint32_t streamId) override;

  void ResetEvent() override;
  void ResetEvent(uint32_t streamId) override;

  void RecordEvent() override;
  void RecordEvent(uint32_t streamId) override;
  bool NeedWait() override;
  void SyncEvent() override;
  bool QueryEvent() override;
  void ElapsedTime(float *costTime, const DeviceEvent *other) override;
  bool DestroyEvent() override;
  void set_wait_stream(aclrtStream waitStream) override { waitStream_ = waitStream; }
  void set_record_stream(aclrtStream recordStream) override { recordStream_ = recordStream; }

 protected:
  aclrtEvent event_{nullptr};
  aclrtStream waitStream_{nullptr};
  aclrtStream recordStream_{nullptr};
  bool needWait_{false};
  bool eventDestroyed_{false};
  bool hasFlag_{false};
};

class MRT_EXPORT AscendTimeEvent : public AscendEvent {
 public:
  AscendTimeEvent();
  ~AscendTimeEvent() override = default;
};
}  // namespace mrt::device::ascend
#endif  // INFERRT_SRC_HARDWARE_ASCEND_ASCEND_EVENT_H
