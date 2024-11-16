#include "token.h"

#include <sstream>

#include "common/common.h"

namespace lexer {
#define TOKEN(T) #T,
const char *_tokens_str[] = {
#include "token_type.list"
};
#undef TOKEN

const char *ToStr(TokenConstPtr token) { return _tokens_str[token->type]; }

std::string ToString(TokenConstPtr token) {
  std::stringstream ss;
  ss << '[' << ToStr(token) << ": ";
  if (token->type == TokenType_Operator) {
    ss << ToStr(token->data.op);
  } else if (token->type == TokenType_Keyword) {
    ss << ToStr(token->data.kw);
  } else if (token->type == TokenType_Separator) {
    ss << ToStr(token->data.sp);
  } else if (token->type == TokenType_Literal) {
    ss << ToStr(token->data.lt);
  } else if (token->type == TokenType_Identifier) {
    ss << token->data.str;
  } else if (token->type == TokenType_Comment) {
    ss << token->data.str;
  } else {
    ss << "?";
  }
  ss << ']';
  return ss.str();
}
} // namespace lexer