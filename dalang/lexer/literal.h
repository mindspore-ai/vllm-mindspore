#ifndef __LEXER_LITERAL_H__
#define __LEXER_LITERAL_H__

#include <cstring>

#define TYPE(T) LiteralId_##T,
typedef enum LiteralId {
  LiteralId_Invalid,
#include "literal_type.list"
  LiteralId_End
} LtId;
#undef TYPE

namespace lexer {
const char *ToStr(LtId lid);
} // namespace lexer

#endif // __LEXER_LITERAL_H__