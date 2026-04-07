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

#ifndef INFERRT_SRC_HARDWARE_ASCEND_ASCEND_CAPTURE_GRAPH_H_
#define INFERRT_SRC_HARDWARE_ASCEND_ASCEND_CAPTURE_GRAPH_H_

#include "acl/acl_mdl.h"
#include "hardware/hardware_abstract/capture_graph.h"

namespace mrt::device::ascend {

class AscendCaptureGraph : public CaptureGraph {
 public:
  AscendCaptureGraph() = default;
  ~AscendCaptureGraph() override;
  bool CaptureBegin(void *stream) override;
  void CaptureGetInfo(void *stream) override;
  void CaptureEnd(void *stream) override;
  void ExecuteCaptureGraph(void *stream) override;
  void CaptureTaskGrpBegin(void *stream) override;
  void CaptureTaskGrpEnd(void *stream, void **task_grp) override;
  void CaptureTaskUpdateBegin(void *updateStream, void *task_grp) override;
  void CaptureTaskUpdateEnd(void *updateStream) override;

 protected:
  aclrtStream capture_stream_{nullptr};
#if defined(__linux__)
  aclmdlRICaptureMode mode_{aclmdlRICaptureMode::ACL_MODEL_RI_CAPTURE_MODE_RELAXED};
  aclmdlRI model_ri_{nullptr};
#endif
  bool finish_capture_graph_{false};
};
}  // namespace mrt::device::ascend
#endif  // INFERRT_SRC_HARDWARE_ASCEND_ASCEND_CAPTURE_GRAPH_H_
