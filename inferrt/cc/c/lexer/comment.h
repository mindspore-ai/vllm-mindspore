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

#ifndef __LEXER_COMMENT_H__
#define __LEXER_COMMENT_H__

namespace da {
namespace lexer {
// Starts with # or //
static inline bool MatchComment(const char *start) {
  int pos = 0;
  if (start[pos] == '\0') {
    return false;
  }
  if (start[pos] == '#') {
    return true;
  }
  if (start[pos + 1] == '\0') {
    return false;
  }
  if (start[pos] == '/' && start[pos + 1] == '/') {
    return true;
  }
  return false;
}
}  // namespace lexer
}  // namespace da

#endif  // __LEXER_COMMENT_H__