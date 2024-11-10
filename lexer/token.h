#ifndef __LEXER_TOKEN_H__
#define __LEXER_TOKEN_H__

#include <string>

#include "lexer/comment.h"
#include "lexer/identifier.h"
#include "lexer/keyword.h"
#include "lexer/literal.h"
#include "lexer/operator.h"
#include "lexer/separator.h"

namespace lexer {
#define TOKEN(T) TokenType_##T,
typedef enum TokenType {
#include "token_type.list"
  TokenType_End,
} TokenType;
#undef TOKEN

typedef struct Token {
  TokenType type;
  union {
    KwId kw;
    SpId sp;
    OpId op;
    const char *name; // For identifier, literal, and comment.
  } data;             // Token detail in specific type.
  int lineStart;
  int lineEnd;
  int columnStart;
  int columnEnd;
  std::string name;

  bool IsSeparatorSpace() const {
    return type == TokenType_Separator && data.sp == SpId_Space;
  }
} Token;
typedef Token *TokenPtr;
typedef const Token *TokenConstPtr;

Token TraverseOpTable(const char *start);
Token TraverseKwTable(const char *start);
Token TraverseSpTable(const char *start);
Token FindLiteral(const char *start);
Token FindIdentifier(const char *start);
Token FindComment(const char *start, size_t len);

const char *ToStr(TokenConstPtr token);
} // namespace lexer

#endif // __LEXER_TOKEN_H__