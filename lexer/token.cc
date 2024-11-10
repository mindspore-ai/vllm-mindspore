#include "token.h"
#include "common/common.h"

namespace lexer {
#define TOKEN(T) #T,
const char *_tokens_str[] = {
#include "token_type.list"
};
#undef TOKEN

const char *ToStr(TokenConstPtr token) { return _tokens_str[token->type]; }
} // namespace lexer