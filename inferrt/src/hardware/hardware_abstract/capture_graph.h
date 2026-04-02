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

#ifndef INFERRT_SRC_HARDWARE_CAPTURE_GRAPH_H
#define INFERRT_SRC_HARDWARE_CAPTURE_GRAPH_H

#include <memory>
#include <vector>

namespace mrt {
class CaptureGraph {
 public:
  virtual ~CaptureGraph() = default;
  virtual bool CaptureBegin(void *stream) = 0;
  virtual void CaptureGetInfo(void *stream) = 0;
  virtual void CaptureEnd(void *stream) = 0;
  virtual void ExecuteCaptureGraph(void *stream) = 0;
  virtual void CaptureTaskGrpBegin(void *stream) = 0;
  virtual void CaptureTaskGrpEnd(void *stream, void **task_grp) = 0;
  virtual void CaptureTaskUpdateBegin(void *updateStream, void *task_grp) = 0;
  virtual void CaptureTaskUpdateEnd(void *updateStream) = 0;
};
using CaptureGraphPtr = std::shared_ptr<CaptureGraph>;
}  // namespace mrt
#endif  // INFERRT_SRC_HARDWARE_CAPTURE_GRAPH_H
