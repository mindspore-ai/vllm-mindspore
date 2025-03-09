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
#include "ops/ops_name.h"

#include <unordered_map>

#include "common/logger.h"

namespace ops {
#define OP(O) {#O, Op_##O},
std::unordered_map<std::string_view, Op> _opNames{
#include "ops/ops.list"
};
#undef OP

Op MatchOp(const char *op) {
  if (_opNames.count(op) == 0) {
    LOG_ERROR << "Not found op with name '" << op << "'";
    for (auto it = _opNames.cbegin(); it != _opNames.cend(); ++it) {
      LOG_OUT << "#" << it->first << ", " << it->second;
    }
    exit(EXIT_FAILURE);
  }
  return _opNames[op];
}

#define OP(O) #O,
const char *_opStr[] = {
#include "ops/ops.list"
    "End",
};
#undef OP

const char *ToStr(const Op op) { return _opStr[op]; }
} // namespace ops
