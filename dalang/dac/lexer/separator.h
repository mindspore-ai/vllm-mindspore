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

#ifndef __LEXER_SEPARATOR_H__
#define __LEXER_SEPARATOR_H__

namespace da {
namespace lexer {
typedef enum SeparatorId {
  SpId_Space,            // sp
  SpId_Tab,              // tab
  SpId_LeftParenthesis,  // (
  SpId_RightParenthesis, // )
  SpId_LeftBracket,      // [
  SpId_RightBracket,     // ]
  SpId_LeftBrace,        // {
  SpId_RightBrace,       // }
  SpId_Semicolon,        // ;
  SpId_Comma,            // ,
  SpId_Dot,              // .
  SpId_Colon,            // :
  SpId_Question,         // ?
  SpId_Pound,            // #
  SpId_End,
} SpId;

typedef struct NameToSeparatorId {
  const char *name;
  SpId id;
} NameToSpId;

const char *ToStr(SpId spid);
} // namespace lexer
} // namespace da

#endif // __LEXER_SEPARATOR_H__