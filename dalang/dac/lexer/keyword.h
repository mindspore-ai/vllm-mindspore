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

#ifndef __LEXER_KEYWORD_H__
#define __LEXER_KEYWORD_H__

namespace lexer {
#define KEYWORD(K) KwId_##K,
#define KEYWORD_ALIAS(K, ALIAS) KwId_##K,
#define KEYWORD_ALIAS2(K, ALIAS1, ALIAS2) KwId_##K,
typedef enum KeywordId {
#include "keyword.list"
  KwId_End
} KwId;
#undef KEYWORD
#undef KEYWORD_ALIAS
#undef KEYWORD_ALIAS2

typedef struct NameToKeywordId {
  const char *name;
  KwId id;
} NameToKwId;

const char *ToStr(KwId kwid);
} // namespace lexer
#endif // __LEXER_KEYWORD_H__