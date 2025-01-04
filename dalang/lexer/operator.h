#ifndef __LEXER_OPERATOR_H__
#define __LEXER_OPERATOR_H__

#define OPERATOR(O) OpId_##O,
typedef enum OperatorId {
#include "lexer/operator.list"
  OpId_End,
} OpId;
#undef OPERATOR

namespace lexer {
typedef struct NameToOperatorId {
  const char *name;
  OpId id;
} NameToOpId;

const char *ToStr(OpId opid);
} // namespace lexer

#endif // __LEXER_OPERATOR_H__