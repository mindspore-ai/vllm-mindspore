#ifndef __PARSER_AST_NODE_H__
#define __PARSER_AST_NODE_H__

#include <vector>

#include "common/common.h"
#include "lexer/token.h"

using namespace lexer;

namespace parser {
typedef struct ExprNode *ExprPtr;
typedef const struct ExprNode *ExprConstPtr;

typedef std::vector<ExprPtr> Exprs;
typedef std::vector<ExprPtr> *ExprsPtr;
typedef const std::vector<ExprPtr> *ExprsConstPtr;

typedef struct StmtNode *StmtPtr;
typedef const struct StmtNode *StmtConstPtr;

typedef std::vector<StmtPtr> Stmts;
typedef std::vector<StmtPtr> *StmtsPtr;
typedef const std::vector<StmtPtr> *StmtsConstPtr;

// Expr pool.
ExprPtr NewExpr();
void ClearExprPool();
ExprsConstPtr ExprList();
void ClearExprListMemory(ExprPtr expr);

// Stmt pool.
StmtPtr NewStmt();
void ClearStmtPool();
StmtsConstPtr StmtList();

// Statement type.
enum StmtType {
  StmtType_Expr,
  StmtType_Assign,
  StmtType_Return,
  StmtType_Function,
  StmtType_Class,
  StmtType_If,
  StmtType_While,
  StmtType_For,
  StmtType_Break,
  StmtType_Continue,
  StmtType_Pass,
  StmtType_Import,
  StmtType_End,
};

// Statement node type.
typedef struct StmtNode {
  StmtType type = StmtType_End;
  union {
    struct {
      ExprConstPtr value{nullptr};
    } Expr;
    struct {
      ExprConstPtr target{nullptr};
      ExprConstPtr value{nullptr};
    } Assign;
    struct {
      ExprConstPtr value{nullptr};
    } Return;
    struct {
      ExprConstPtr name{nullptr};
      ExprConstPtr args{nullptr};
      size_t len{0};
      StmtConstPtr *body{nullptr};
    } Function;
    struct {
      ExprConstPtr name{nullptr};
      ExprConstPtr bases{nullptr};
      size_t len{0};
      StmtConstPtr *body{nullptr};
    } Class;
    struct {
      ExprConstPtr condition{nullptr};
      size_t ifLen{0};
      StmtConstPtr *ifBody{nullptr};
      size_t elseLen{0};
      StmtConstPtr *elseBody{nullptr};
    } If;
    struct {
      ExprConstPtr element{nullptr};
      ExprConstPtr iterator{nullptr};
      StmtConstPtr *body{nullptr};
    } For;
    struct {
      ExprConstPtr condition{nullptr};
      StmtConstPtr *body{nullptr};
    } While;
  } stmt;
  int lineStart;
  int lineEnd;
  int columnStart;
  int columnEnd;
} Stmt;

enum ExprType {
  ExprType_Binary,
  ExprType_Unary,
  ExprType_Attribute,
  ExprType_Subscript,
  ExprType_List,
  ExprType_Call,
  ExprType_Name,
  ExprType_Literal,
  ExprType_End,
};

typedef struct ExprNode {
  ExprType type = ExprType_End;
  union {
    struct {
      OpId op;
      ExprConstPtr left{nullptr};
      ExprConstPtr right{nullptr};
    } Binary;
    struct {
      OpId op;
      ExprConstPtr operand{nullptr};
    } Unary;
    struct {
      const std::string *identifier{nullptr};
    } Name;
    struct {
      LtId kind;
      const std::string *value{nullptr};
    } Literal;
    struct {
      size_t len{0};
      ExprConstPtr *values{nullptr};
    } List;
    struct {
      ExprConstPtr function{nullptr};
      ExprConstPtr list{nullptr};
    } Call;
    struct {
      ExprConstPtr entity{nullptr};
      ExprConstPtr attribute{nullptr};
    } Attribute;
  } expr;
  int lineStart;
  int lineEnd;
  int columnStart;
  int columnEnd;
} Expr;

const std::string ToString(StmtConstPtr stmt);
const std::string ToString(ExprConstPtr expr);
} // namespace parser

#endif // __PARSER_AST_NODE_H__