/**
 * Copyright 2025 Zhang Qinghua
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

#ifndef __OPS_OPS_NAME_H__
#define __OPS_OPS_NAME_H__

namespace ops {
#define OP(O) Op_##O,
enum Op {
#include "ops/ops.list"
  Op_End
};
#undef OP

Op MatchOp(const char *op);
const char *ToStr(Op op);
} // namespace ops

#endif // __OPS_OPS_NAME_H__