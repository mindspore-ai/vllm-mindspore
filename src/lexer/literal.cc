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
    token.data.lt = LiteralId_int; // TODO.
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
    token.name.assign(start, matchEnd - start + 1);
    return token;
  } else if (startStr != '\0') { // String across multiple lines.
    Token token{.type = TokenType_ContinuousString};
    token.data.str = &startStr;
    token.name.assign(start);
    return token;
  } else {
#ifdef DEBUG
    LOG_OUT << "match nothing, " << start << LOG_ENDL;
#endif
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

const char *ToStr(LtId ltid) { return _types_str[ltid]; }
} // namespace lexer