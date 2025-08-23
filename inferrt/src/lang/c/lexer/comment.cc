/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include <unordered_set>

#include "common/common.h"
#include "lang/c/lexer/token.h"

namespace da {
namespace lexer {
Token FindComment(const char *start, size_t len) {
  if (!MatchComment(start)) {
    return Token{.type = TokenType_End};
  }
  Token token{.type = TokenType_Comment};
  token.start = start;
  token.len = len;
  token.name.assign(start, len);
  return token;
}
}  // namespace lexer
}  // namespace da
