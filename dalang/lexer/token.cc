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

#include "token.h"

#include <sstream>

#include "common/common.h"

namespace lexer {
#define TOKEN(T) #T,
const char *_tokens_str[] = {
#include "token_type.list"
    "ContStr",
    "InvalidStr"
    "End",
};
#undef TOKEN

const char *ToStr(TokenConstPtr token) {
  if (token == nullptr) {
    return "Token[null]";
  }
  return _tokens_str[token->type];
}

std::string ToString(TokenConstPtr token) {
  if (token == nullptr) {
    return "Token[null]";
  }
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
  } else if (token->type == TokenType_End) {
    ss << '\'' << token->name << '\'';
  } else {
    ss << "?";
  }
  ss << ']';
  return ss.str();
}
} // namespace lexer