/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#ifndef __LEXER_LEXER_H__
#define __LEXER_LEXER_H__

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "lang/c/lexer/token.h"

namespace da {
namespace lexer {
#define MAX_LINE_SIZE 4096

class Lexer {
 public:
  explicit Lexer(const std::string &filename);
  explicit Lexer(const char *sourceLines);
  ~Lexer();

  const Token NextToken();
  const std::vector<Token> &Tokens();

  void Dump();

  const std::string &filename() const { return filename_; }

  bool supportIndent() const { return supportIndent_; }

 private:
  void OpenFile(const std::string &filename);
  void ReadLine();
  const Token TokenInLine();
  bool IsLineEnd() const;
  void SetLineInfo(TokenPtr token);

  Token GetOperator();
  Token GetSeparator();
  Token GetKeyword();
  Token GetLiteral();
  Token GetIdentifier();
  Token GetComment();

  bool HandleNewLineIndent();

  std::string UnescapeString(const std::string &str);
  std::string EscapeString(const std::string &str);

  // Read from file.
  std::string filename_;
  std::ifstream sourceFileStream_;
  // Read source directly.
  const char *sources_;
  std::istringstream sourceStringStream_;

  std::string line_;
  size_t lineno_{0};
  size_t column_{0};
  bool eof_{false};
  bool skipWhiteSpace_{false};
  bool scanned_{false};
  bool supportIndent_{true};
  std::vector<std::string> indents_;
  std::vector<Token> tokens_;
};
}  // namespace lexer
}  // namespace da
#endif  // __LEXER_LEXER_H__
