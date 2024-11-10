#include <unordered_set>

#include "common/common.h"
#include "token.h"

namespace lexer {
std::unordered_set<char> _decimal = {
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
};
std::unordered_set<char> _hexadecimal = {'0', '1', '2', '3', '4', '5', '6', '7',
                                         '8', '9', 'a', 'b', 'c', 'd', 'e', 'f',
                                         'A', 'B', 'C', 'D', 'E', 'F'};

std::unordered_set<char> _alphabets = {
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
};

static inline int Char2Int(char c) {
  if (c >= '0' && c <= '9') {
    return c - '0';
  } else if (c >= 'a' && c <= 'f') {
    return c - 'a' + 10;
  } else if (c >= 'A' && c <= 'F') {
    return c - 'A' + 10;
  } else {
    LOG_ERROR << "Unsupported char in Char2Int().";
    exit(-1);
  }
}

Token FindLiteral(const char *start) {
  // Decimal digital
  int pos = MatchDecimal(start);
  if (pos != 0) {
    Token token{.type = TokenType_Literal};
    token.name.assign(start, pos);
    return token;
  }
  // String wrapped by ' or "
  pos = MatchString(start);
  if (pos != 0) {
    Token token{.type = TokenType_Literal};
    token.name.assign(start, pos);
    return token;
  }
  return Token{.type = TokenType_End};
}

#define TYPE(T) #T,
const char *_types_str[] = {
    "Invalid",
#include "literal_type.list"
    "End",
};
#undef TYPE

const char *ToStr(LiteralId lid) { return _types_str[lid]; }
} // namespace lexer