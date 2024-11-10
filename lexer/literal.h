#ifndef __LEXER_LITERAL_H__
#define __LEXER_LITERAL_H__

#define TYPE(K) LiteralId_##K,
typedef enum LiteralId {
  LiteralId_Invalid,
#include "literal_type.list"
  LiteralId_End
} LiteralId;
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

static inline int MatchString(const char *start) {
  int pos = 0;
  char c = start[pos];
  if (c == '\0') {
    return 0;
  }
  if (c == '\'' || c == '\"') {
    ++pos;
    while (start[pos] != '\0' && start[pos] != c) {
      ++pos;
    }
    if (start[pos] == c) { // Found a string.
      return pos + 1;
    }
  }
  return 0;
}

const char *ToStr(LiteralId lid);
} // namespace lexer

#endif // __LEXER_LITERAL_H__