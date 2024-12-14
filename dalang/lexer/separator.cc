#include "common/common.h"
#include "token.h"

namespace lexer {
NameToSpId _separators[] = {
    {"Invalid", SpId_Invalid},
    {" ", SpId_Space},
    {"(", SpId_LeftParenthesis},
    {")", SpId_RightParenthesis},
    {"[", SpId_LeftBracket},
    {"]", SpId_RightBracket},
    {"{", SpId_LeftBrace},
    {"}", SpId_RightBrace},
    {";", SpId_Semicolon},
    {",", SpId_Comma},
    {".", SpId_Dot},
    {":", SpId_Colon},
    {"?", SpId_Question},
    {"#", SpId_Pound},
    {"End", SpId_End},
};

Token TraverseSpTable(const char *start) {
  auto pos = FindNameIndex<NameToSpId>(
      start, _separators, sizeof(_separators) / sizeof(NameToSpId));
  if (pos != -1) {
    const auto &sp = _separators[pos];
    auto t = Token{.type = TokenType_Separator};
    t.data.sp = sp.id;
    t.start = start;
    t.len = strlen(sp.name);
    t.name.assign(sp.name, strlen(sp.name));
    return t;
  }
  return Token{.type = TokenType_End};
}

const char *ToStr(SpId spid) { return _separators[spid].name; }
} // namespace lexer