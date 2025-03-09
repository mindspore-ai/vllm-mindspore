/**
 * Copyright 2024 Zhang Qinghua
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

#include "common/common.h"
#include "token.h"

namespace lexer {
#define TO_STR(s) #s
#define KEYWORD(K) {TO_STR(K), KwId_##K},
NameToKwId _keywords[]{
#include "keyword.list"
};
#undef KEYWORD

Token TraverseKwTable(const char *start) {
  auto pos = FindNameIndex<NameToKwId>(start, _keywords,
                                       sizeof(_keywords) / sizeof(NameToKwId));
  if (pos != -1) {
    const auto &kw = _keywords[pos];
    auto t = Token{.type = TokenType_Keyword};
    t.data.kw = kw.id;
    t.start = start;
    t.len = strlen(kw.name);
    t.name.assign(kw.name, strlen(kw.name));
    return t;
  }
  return Token{.type = TokenType_End};
}

#define KEYWORD(T) #T,
const char *_keywordsStr[] = {
#include "keyword.list"
    "End",
};
#undef KEYWORD

const char *ToStr(KwId kwid) { return _keywordsStr[kwid]; }
} // namespace lexer