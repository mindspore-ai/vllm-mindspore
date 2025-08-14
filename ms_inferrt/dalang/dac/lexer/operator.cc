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

#include "common/common.h"
#include "token.h"

namespace da {
namespace lexer {
NameToOpId _operators[] = {
  {"==", OpId_Equal},     {"!=", OpId_NotEqual},   {"<=", OpId_LessEqual},  {">=", OpId_GreaterEqual},
  {"||", OpId_LogicalOr}, {"&&", OpId_LogicalAnd}, {">>", OpId_ShiftRight}, {"<<", OpId_ShiftLeft},
  {">:", OpId_StdCin},    {"<:", OpId_StdCout},    {"<", OpId_LessThan},    {">", OpId_GreaterThan},
  {"+=", OpId_AddAssign}, {"-=", OpId_SubAssign},  {"*=", OpId_MulAssign},  {"/=", OpId_DivAssign},
  {"%=", OpId_ModAssign}, {"=", OpId_Assign},      {"+", OpId_Add},         {"-", OpId_Sub},
  {"*", OpId_Mul},        {"/", OpId_Div},         {"%", OpId_Mod},
};

Token TraverseOpTable(const char *start) {
  auto pos = FindNameIndex<NameToOpId>(start, _operators, sizeof(_operators) / sizeof(NameToOpId));
  if (pos != -1) {
    const auto &op = _operators[pos];
    auto t = Token{.type = TokenType_Operator};
    t.data.op = op.id;
    t.start = start;
    t.len = strlen(op.name);
    t.name.assign(op.name, strlen(op.name));
    return t;
  }
  return Token{.type = TokenType_End};
}

#define OPERATOR(T) #T,
const char *_operatorsStr[] = {
#include "operator.list"
  "End",
};
#undef OPERATOR

const char *ToStr(OpId opid) { return _operatorsStr[opid]; }
}  // namespace lexer
}  // namespace da