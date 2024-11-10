#include <unordered_set>

#include "common/common.h"
#include "token.h"

namespace lexer {
Token FindComment(const char *start, size_t len) {
  if (!MatchComment(start)) {
    return Token{.type = TokenType_End};
  }
  Token token{.type = TokenType_Comment};
  token.name.assign(start, len);
  return token;
}
} // namespace lexer