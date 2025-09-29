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

#ifndef __OPS_ASCEND_ACLNN_UTILS_HASH_BUF_H__
#define __OPS_ASCEND_ACLNN_UTILS_HASH_BUF_H__

#include <cstdint>

#include "common/common.h"
#include "include/securec.h"

namespace mrt {
namespace ops {
inline constexpr int gHashBufSize = 8192;
inline constexpr int gHashBufMaxSize = gHashBufSize + 1024;
extern thread_local char gHashBuf[gHashBufSize];
extern thread_local int gHashOffset;

inline void MemcpyToBuf(const void *data, size_t size) {
  if (size == 0) {
    return;
  }
  if (MS_UNLIKELY(static_cast<uint64_t>(gHashOffset) > SIZE_MAX - size)) {
    LOG_ERROR << "Hash buf is overflow.";
    return;
  }
  if (gHashOffset + size >= gHashBufSize) {
    gHashOffset = gHashBufMaxSize;
    return;
  }
  auto ret = memcpy_sp(gHashBuf + gHashOffset, gHashBufSize - gHashOffset, data, size);
  if (ret != EOK) {
    LOG_EXCEPTION << "Failed to memcpy!";
  }
  gHashOffset += size;
}

uint64_t GenHash(const void *key, const int len, const uint32_t seed = 0xdeadb0d7);

inline uint64_t CalcHashId() {
  if (gHashOffset == gHashBufMaxSize) {
    return 0;
  }
  uint64_t hashId = GenHash(gHashBuf, gHashOffset);
  return hashId;
}

}  // namespace ops
}  // namespace mrt
#endif  // __OPS_ASCEND_ACLNN_UTILS_HASH_BUF_H__
