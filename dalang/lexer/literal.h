#ifndef __LEXER_LITERAL_H__
#define __LEXER_LITERAL_H__

#define TYPE(K) LiteralId_##K,
typedef enum LiteralId {
  LiteralId_Invalid,
#include "literal_type.list"
  LiteralId_End
} LtId;
#undef TYPE

namespace lexer {
static inline int MatchDecimal(const char *start) {
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

static inline const char *MatchString(const char *start, char *startChar) {
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

const char *ToStr(LtId lid);
} // namespace lexer

#endif // __LEXER_LITERAL_H__