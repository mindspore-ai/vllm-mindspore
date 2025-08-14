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

#ifndef __LEXER_IDENTIFIER_H__
#define __LEXER_IDENTIFIER_H__

namespace da {
namespace lexer {
// Identifier name like c/cpp.
static inline int MatchName(const char *start) {
  int pos = 0;
  char c = start[pos];
  if (c == '\0') {
    return 0;
  }
  // Starts with [a-z][A-Z]_
  if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_') {
    ++pos;
    // Follow with [a-z][A-Z][0-9]_
    while (start[pos] != '\0') {
      c = start[pos];
      if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_') {
        ++pos;
      } else {
        return pos;
      }
    }
  }
  return pos;
}
}  // namespace lexer
}  // namespace da

#endif  // __LEXER_IDENTIFIER_H__