#include <unordered_set>

#include "common/common.h"
#include "token.h"

namespace lexer {
Token FindIdentifier(const char *start) {
  int pos = MatchName(start);
  if (pos == 0) {
    return Token{.type = TokenType_End};
  }
  Token token{.type = TokenType_Identifier};
  token.start = start;
  token.len = pos;
  token.name.assign(start, pos);
  return token;
}
} // namespace lexer