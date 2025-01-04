#ifndef __LEXER_KEYWORD_H__
#define __LEXER_KEYWORD_H__

namespace lexer {
#define KEYWORD(K) KwId_##K,
typedef enum KeywordId {
#include "keyword.list"
  KwId_End
} KwId;
#undef KEYWORD

typedef struct NameToKeywordId {
  const char *name;
  KwId id;
} NameToKwId;

const char *ToStr(KwId kwid);
} // namespace lexer
#endif // __LEXER_KEYWORD_H__