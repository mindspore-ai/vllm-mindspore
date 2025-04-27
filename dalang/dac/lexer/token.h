/**
 * Copyright 2024 Zhang Qinghua
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __LEXER_TOKEN_H__
#define __LEXER_TOKEN_H__

#include <string>

#include "lexer/comment.h"
#include "lexer/identifier.h"
#include "lexer/keyword.h"
#include "lexer/literal.h"
#include "lexer/operator.h"
#include "lexer/separator.h"

namespace da {
namespace lexer {
#define TOKEN(T) TokenType_##T,
typedef enum TokenType {
#include "token_type.list"
  TokenType_ContinuousString,
  TokenType_InvalidString,
  TokenType_End,
} TokenType;
#undef TOKEN

typedef struct Token {
  TokenType type;
  const char *start;
  size_t len;
  union {
    KwId kw;
    SpId sp;
    OpId op;
    LtId lt;
    const char *str; // For identifier, and comment.
  } data;            // Token detail in specific type.
  int lineStart;
  int lineEnd;
  int columnStart;
  int columnEnd;
  std::string name;

  bool IsSeparatorSpace() const {
    return type == TokenType_Separator &&
           (data.sp == SpId_Space || data.sp == SpId_Tab);
  }
  bool IsIndentBlockStart() const {
    return type == TokenType_Separator && data.sp == SpId_Colon;
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
std::string ToString(TokenConstPtr token);
} // namespace lexer
} // namespace da

#endif // __LEXER_TOKEN_H__