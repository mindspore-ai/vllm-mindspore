#ifndef __LEXER_COMMENT_H__
#define __LEXER_COMMENT_H__

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
} // namespace lexer

#endif // __LEXER_COMMENT_H__