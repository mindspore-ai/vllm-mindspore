#ifndef __LEXER_SEPARATOR_H__
#define __LEXER_SEPARATOR_H__

namespace lexer {
typedef enum SeparatorId {
  SpId_Invalid,
  SpId_Space,
  SpId_LeftParenthesis,
  SpId_RightParenthesis,
  SpId_LeftBracket,
  SpId_RightBracket,
  SpId_LeftBrace,
  SpId_RightBrace,
  SpId_Semicolon,
  SpId_Comma,
  SpId_Dot,
  SpId_Colon,
  SpId_Question,
  SpId_Pound,
  SpId_End,
} SpId;

typedef struct NameToSeparatorId {
  const char *name;
  SpId id;
} NameToSpId;
} // namespace lexer

#endif // __LEXER_SEPARATOR_H__