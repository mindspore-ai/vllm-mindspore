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

#ifndef __PARSER_AST_NODE_H__
#define __PARSER_AST_NODE_H__

#include <vector>

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
#define STMT(type) StmtType_##type,
enum StmtType {
#include "stmt.list"
  StmtType_End,
};
#undef STMT

// Statement node type.
typedef struct StmtNode {
  StmtType type = StmtType_End;
  union {
    struct {
      size_t len{0};
      StmtConstPtr *body{nullptr};
    } Module;
    struct {
      ExprConstPtr value{nullptr};
    } Expr;
    struct {
      ExprConstPtr target{nullptr};
      ExprConstPtr value{nullptr};
    } Assign;
    struct {
      ExprConstPtr target{nullptr};
      OpId op;
      ExprConstPtr value{nullptr};
    } AugAssign;
    struct {
      ExprConstPtr value{nullptr};
    } Return;
    struct {
      ExprConstPtr name{nullptr};
      size_t argsLen{0};
      StmtConstPtr *args{nullptr};
      size_t len{0};
      StmtConstPtr *body{nullptr};
    } Graph;
    struct {
      ExprConstPtr name{nullptr};
      size_t argsLen{0};
      StmtConstPtr *args{nullptr};
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
      size_t len{0};
      StmtConstPtr *body{nullptr};
    } Block;
    struct {
      ExprConstPtr value{nullptr};
    } StdCin;
    struct {
      ExprConstPtr value{nullptr};
    } StdCout;
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
      size_t len{0};
      StmtConstPtr *body{nullptr};
    } For;
    struct {
      ExprConstPtr condition{nullptr};
      size_t len{0};
      StmtConstPtr *body{nullptr};
    } While;
  } stmt;
  int lineStart{-1};
  int lineEnd{-1};
  int columnStart{-1};
  int columnEnd{-1};
} Stmt;

#define EXPR(type) ExprType_##type,
enum ExprType {
#include "expr.list"
  ExprType_End,
};
#undef EXPR

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