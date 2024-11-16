#include "common/common.h"
#include "token.h"

namespace lexer {
NameToOpId _operators[] = {
    {"==", OpId_Equal},        {"!=", OpId_NotEqual},  {"<=", OpId_LessEqual},
    {">=", OpId_GreaterEqual}, {"||", OpId_LogicalOr}, {"&&", OpId_LogicalAnd},
    {">>", OpId_ShiftRight},   {"<<", OpId_ShiftLeft}, {"<", OpId_LessThan},
    {">", OpId_GreaterThan},   {"+", OpId_Add},        {"-", OpId_Sub},
    {"*", OpId_Mul},           {"/", OpId_Div},        {"=", OpId_Assign},
};

Token TraverseOpTable(const char *start) {
  auto pos = FindNameIndex<NameToOpId>(start, _operators,
                                       sizeof(_operators) / sizeof(NameToOpId));
  if (pos != -1) {
    const auto &op = _operators[pos];
    auto t = Token{.type = TokenType_Operator};
    t.data.op = op.id;
    t.name.assign(op.name, strlen(op.name));
    return t;
  }
  return Token{.type = TokenType_End};
}

#define OPERATOR(T) #T,
const char *_operators_str[] = {
    "Invalid",
#include "operator.list"
    "End",
};
#undef OPERATOR

const char *ToStr(OpId opid) { return _operators_str[opid]; }
} // namespace lexer