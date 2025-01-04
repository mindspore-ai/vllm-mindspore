#include "common/common.h"
#include "token.h"

namespace lexer {
#define TO_STR(s) #s
#define KEYWORD(K) {TO_STR(K), KwId_##K},
NameToKwId _keywords[]{
#include "keyword.list"
};
#undef KEYWORD

Token TraverseKwTable(const char *start) {
  auto pos = FindNameIndex<NameToKwId>(start, _keywords,
                                       sizeof(_keywords) / sizeof(NameToKwId));
  if (pos != -1) {
    const auto &kw = _keywords[pos];
    auto t = Token{.type = TokenType_Keyword};
    t.data.kw = kw.id;
    t.start = start;
    t.len = strlen(kw.name);
    t.name.assign(kw.name, strlen(kw.name));
    return t;
  }
  return Token{.type = TokenType_End};
}

#define KEYWORD(T) #T,
const char *_keywords_str[] = {
#include "keyword.list"
    "End",
};
#undef KEYWORD

const char *ToStr(KwId kwid) { return _keywords_str[kwid]; }
} // namespace lexer