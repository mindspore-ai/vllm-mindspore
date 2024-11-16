#ifndef __LEXER_LEXER_H__
#define __LEXER_LEXER_H__

#include <fstream>
#include <string>
#include <vector>

#include "common/common.h"
#include "token.h"

namespace lexer {
#define MAX_LINE_SIZE 4096

class Lexer {
public:
  explicit Lexer(const std::string &filename);
  ~Lexer();

  const Token NextToken();
  const std::vector<Token> &Tokens();

private:
  void OpenFile(const std::string &filename);
  const std::string &ReadLine();
  const Token TokenInLine();
  bool IsLineEnd() const;
  void SetLineInfo(TokenPtr token);

  Token GetOperator();
  Token GetSeparator();
  Token GetKeyword();
  Token GetLiteral();
  Token GetIdentifier();
  Token GetComment();

  std::string filename_;
  std::ifstream file_;
  std::string line_;
  size_t lineno_{0};
  size_t pos_{0};
  bool eof_{false};
  bool skipWhiteSpace_{false};
  bool scanned_{false};
  std::vector<Token> tokens_;
};
} // namespace lexer
#endif // __LEXER_LEXER_H__