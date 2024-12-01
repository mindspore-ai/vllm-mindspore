#ifndef __PARSER_IR_OPERATOR_H__
#define __PARSER_IR_OPERATOR_H__

#include <vector>

namespace ir {
#define OPERATOR(O) Op_##O,
typedef enum Operator {
  Op_Invalid,
#include "lexer/operator.list"
  Op_End,
} Op;
#undef OPERATOR

} // namespace ir

#endif // __PARSER_IR_OPERATOR_H__