#ifndef __LEXER_IDENTIFIER_H__
#define __LEXER_IDENTIFIER_H__

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
      if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
          (c >= '0' && c <= '9') || c == '_') {
        ++pos;
      } else {
        return pos;
      }
    }
  }
  return pos;
}
} // namespace lexer

#endif // __LEXER_IDENTIFIER_H__