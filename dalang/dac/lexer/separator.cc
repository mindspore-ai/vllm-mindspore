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
NameToSpId _separators[] = {
    {" ", SpId_Space},
    {"(", SpId_LeftParenthesis},
    {")", SpId_RightParenthesis},
    {"[", SpId_LeftBracket},
    {"]", SpId_RightBracket},
    {"{", SpId_LeftBrace},
    {"}", SpId_RightBrace},
    {";", SpId_Semicolon},
    {",", SpId_Comma},
    {".", SpId_Dot},
    {":", SpId_Colon},
    {"?", SpId_Question},
    {"#", SpId_Pound},
    {"End", SpId_End},
};

Token TraverseSpTable(const char *start) {
  auto pos = FindNameIndex<NameToSpId>(
      start, _separators, sizeof(_separators) / sizeof(NameToSpId));
  if (pos != -1) {
    const auto &sp = _separators[pos];
    auto t = Token{.type = TokenType_Separator};
    t.data.sp = sp.id;
    t.start = start;
    t.len = strlen(sp.name);
    t.name.assign(sp.name, strlen(sp.name));
    return t;
  }
  return Token{.type = TokenType_End};
}

const char *ToStr(SpId spid) { return _separators[spid].name; }
} // namespace lexer