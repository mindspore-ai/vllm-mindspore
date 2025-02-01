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

#include <unordered_set>

#include "common/common.h"
#include "common/logger.h"
#include "token.h"

namespace lexer {
#define TYPE(T) #T,
const char *_types_str[] = {
#include "literal_type.list"
    "End",
};
#undef TYPE

const char *ToStr(LtId ltid) { return _types_str[ltid]; }

std::unordered_set<char> _decimal = {
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
};
std::unordered_set<char> _hexadecimal = {'0', '1', '2', '3', '4', '5', '6', '7',
                                         '8', '9', 'a', 'b', 'c', 'd', 'e', 'f',
                                         'A', 'B', 'C', 'D', 'E', 'F'};

std::unordered_set<char> _alphabets = {
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
};

namespace {
int Char2Int(char c) {
  if (c >= '0' && c <= '9') {
    return c - '0';
  } else if (c >= 'a' && c <= 'f') {
    return c - 'a' + 10;
  } else if (c >= 'A' && c <= 'F') {
    return c - 'A' + 10;
  } else {
    LOG_ERROR << "Unsupported char in Char2Int().";
    exit(EXIT_FAILURE);
  }
}

size_t StartsWithLiteralType(const char *literal, size_t *count) {
  for (size_t i = 0; i < LiteralId_End; ++i) {
    if (strstr(literal, _types_str[i]) == literal) {
      *count = strlen(_types_str[i]);
      return i;
    }
  }
  *count = 0;
  return 0;
}

LtId MatchLiteralType(const char *start, size_t *matchCount) {
  int pos = SkipWhiteSpace(start);
  char c = start[pos];
  if (c != ':') {
    // No literal type, set int type in default.
    return LiteralId_int;
  }
  ++pos; // ':'
  pos += SkipWhiteSpace(start + pos);
  // Start to match literal type string.
  size_t count;
  auto matchIndex = StartsWithLiteralType(start + pos, &count);
  if (count == 0 || matchIndex >= LiteralId_End) {
    return LiteralId_End;
  }
  pos += count;
  *matchCount = pos;
  return (LtId)matchIndex;
}

int MatchBoolean(const char *start) {
  int pos = 0;
  const char *matchPos = strstr(start, "true");
  if (matchPos == start) {
    return 4;
  }
  matchPos = strstr(start, "false");
  if (matchPos == start) {
    return 5;
  }
  return pos;
}

int MatchDecimal(const char *start) {
  int pos = 0;
  while (start[pos] != '\0') {
    char c = start[pos];
    if (c >= '0' && c <= '9') {
      ++pos;
    } else {
      return pos;
    }
  }
  return pos;
}

const char *MatchString(const char *start, char *startChar) {
  int pos = 0;
  char c = start[pos];
  if (c == '\0') {
    return nullptr;
  }
  if (c == '\'' || c == '\"') {
    *startChar = c;
    ++pos;
    while (start[pos] != '\0' && start[pos] != c) {
      ++pos;
    }
    if (start[pos] == c) { // Found a string.
      return start + pos;
    }
  }
  return nullptr;
}
} // namespace

Token FindLiteral(const char *start) {
  // Boolean
  int pos = MatchBoolean(start);
  if (pos != 0) {
    Token token{.type = TokenType_Literal};
    token.data.lt = LiteralId_bool;
    token.start = start;
    token.len = pos;
    token.name.assign(start, pos);
    return token;
  }
  // Decimal digital
  pos = MatchDecimal(start);
  if (pos != 0) {
    Token token{.type = TokenType_Literal};
    size_t count = 0;
    LtId li = MatchLiteralType(start + pos, &count);
    if (li == LiteralId_End) {
      return Token{
          .type = TokenType_InvalidString, .start = start + pos, .len = 0};
    }
    token.data.lt = li;
    token.start = start;
    token.len = pos + count;
    token.name.assign(start, pos);
    return token;
  }
  // String wrapped by ' or "
  static char startStr;
  startStr = '\0';
  const char *matchEnd = MatchString(start, &startStr);
  if (matchEnd != nullptr) {
    Token token{.type = TokenType_Literal};
    token.data.lt = LiteralId_str;
    const auto strStart = start + 1;         // No ' or ".
    const auto strLen = matchEnd - strStart; // No ' or ".
    token.start = strStart;
    token.len = strLen;
    token.name.assign(strStart, strLen);
    return token;
  } else if (startStr != '\0') { // String across multiple lines.
    Token token{.type = TokenType_ContinuousString};
    token.data.str = &startStr;
    const auto strStart = start + 1; // No ' or ".
    token.start = strStart;
    token.len = strlen(strStart);
    token.name.assign(strStart);
    return token;
  } else {
#ifdef DEBUG
    LOG_OUT << "match nothing, " << start;
#endif
  }
  return Token{.type = TokenType_End, .start = start, .len = 0};
}
} // namespace lexer