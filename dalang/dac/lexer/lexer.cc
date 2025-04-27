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

#include "lexer.h"

#include <cassert>
#include <fstream>
#include <sstream>

#include "common/common.h"
#include "common/logger.h"

namespace da {
namespace lexer {
Lexer::Lexer(const std::string &filename)
    : filename_{filename}, sourceFileStream_{std::ifstream()} {
  LOG_OUT << "filename: " << filename;
  OpenFile(filename);
}

Lexer::Lexer(const char *sourceLines)
    : sources_{sourceLines}, sourceStringStream_{sourceLines} {
  LOG_OUT << "sourceLines: " << sourceLines;
}

Lexer::~Lexer() {
  if (sourceFileStream_.is_open()) {
    sourceFileStream_.close();
  }
  LOG_OUT << "Call ~Lexer";
}

namespace {
const Token MakeIndentFinishToken() {
  // Insert a phony } separator for indent change.
  auto tok = Token{.type = TokenType_Separator};
  tok.data.sp = SpId_RightBrace;
  tok.name = "}[PHONY]";
  LOG_OUT << "Insert a phony } separator for indent decreasing";
  return tok;
}
} // namespace

const Token Lexer::NextToken() {
  // Skip blank line.
  while (IsLineEnd()) {
    if (eof_) {
      // Support indent feature. Handle the last line.
      if (supportIndent_ && !indents_.empty()) {
        indents_.clear();
        return MakeIndentFinishToken();
      }

      LOG_OUT << "No line any more";
      return Token{.type = TokenType_End};
    }
    ReadLine();

    // Support indent feature. Check the indent when get the new line.
    if (supportIndent_ && HandleNewLineIndent()) {
      return MakeIndentFinishToken();
    }
  }
  return TokenInLine();
}

const std::vector<Token> &Lexer::Tokens() {
  if (scanned_) {
    return tokens_;
  }
  for (EVER) {
    auto token = NextToken();
    if (token.type == TokenType_End) {
      LOG_OUT << "No token anymore";
      break;
    }
    if (token.IsSeparatorSpace()) {
      continue;
    }
    LOG_OUT << "# token: " << token.name << "\t\t\t[" << ToStr(&token) << "]";
    tokens_.emplace_back(std::move(token));
  }
  scanned_ = true;
  return tokens_;
}

// If block end, return true.
bool Lexer::HandleNewLineIndent() {
  column_ += SkipWhiteSpace(line_.c_str() + column_);
  if (column_ == 0 && indents_.empty()) { // No indent at all.
    // Ignore.
    LOG_OUT << "No indent at all, column_: " << column_;
  } else if (column_ == 0 && !indents_.empty()) { // No indent, but
                                                  // there're previous indents.
    indents_.clear();
    LOG_OUT << "Block end, column_: " << column_;
    return true;
  } else if (column_ != 0 && indents_.empty()) { // New indent, and no previous
                                                 // indent. Just record indent.
    const auto &currentTok = tokens_.back();
    if (currentTok.IsIndentBlockStart()) {
      std::string indent = std::string(line_, 0, column_);
      LOG_OUT << "New block start, " << indent.size()
              << ", column_: " << column_;
      indents_.emplace_back(std::move(indent));
    }
  } else if (column_ != 0 &&
             !indents_.empty()) { // New indent, and there's previous indents.
    // Compare the indent and previous indent firstly.
    std::string indent = std::string(line_, 0, column_);
    auto lastIndentLen = indents_.back().size();
    auto currentIndentLen = indent.size();
    LOG_OUT << "lastIndentLen: " << lastIndentLen
            << ", currentIndentLen: " << currentIndentLen
            << ", column_: " << column_;
    if (lastIndentLen > currentIndentLen) { // Block end.
      indents_.pop_back();
      LOG_OUT << "Block end, " << lastIndentLen << ", " << currentIndentLen
              << ", column_: " << column_;
      return true;
    } else if (lastIndentLen < currentIndentLen) { // New block.
                                                   // Record the indent.
      const auto &currentTok = tokens_.back();
      if (currentTok.IsIndentBlockStart()) {
        LOG_OUT << "New block start, " << lastIndentLen << ", "
                << currentIndentLen << ", column_: " << column_;
        indents_.emplace_back(std::move(indent));
      }
    } else { // Same indent, ignore.
      LOG_OUT << "Same indent: '" << indent << "', " << lastIndentLen << ", "
              << currentIndentLen << ", column_: " << column_;
    }
  }
  return false;
}

const Token Lexer::TokenInLine() {
  if (skipWhiteSpace_) {
    column_ += SkipWhiteSpace(line_.c_str() + column_);
    if (IsLineEnd()) {
      return Token{.type = TokenType_End};
    }
  }

  Token token = GetComment();
  if (token.type != TokenType_End) {
    assert(token.len == token.name.size());
    column_ += token.len;
    return token;
  }
  token = GetOperator();
  if (token.type != TokenType_End) {
    assert(token.len == token.name.size());
    column_ += token.len;
    return token;
  }
  token = GetSeparator();
  if (token.type != TokenType_End) {
    assert(token.len == token.name.size());
    column_ += token.len;
    return token;
  }
  token = GetKeyword();
  if (token.type != TokenType_End) {
    assert(token.len == token.name.size());
    column_ += token.len;
    return token;
  }
  token = GetLiteral();
  if (token.type != TokenType_End) {
    if (token.lineStart == token.lineEnd) { // Not multiple lines.
      assert(token.len >= token.name.size());
      column_ += token.len;
      if (token.data.lt == LiteralId_str) {
        column_ += 2; // Swallow two ' or " for string literal.
      }
    }
    if (token.data.lt == LiteralId_str) {
      token.name = std::move(UnescapeString(token.name));
    }
    return token;
  }
  token = GetIdentifier();
  if (token.type != TokenType_End) {
    assert(token.len == token.name.size());
    column_ += token.len;
    return token;
  }

  // Not match any excepted token, return a invalid token and move on.
  LOG_OUT << "Not match any excepted token, line: " << line_
          << ", column: " << column_;
  Token invalidToken = Token{.type = TokenType_End};
  if (!IsLineEnd()) {
    invalidToken.name = line_.at(column_);
  }
  SetLineInfo(&invalidToken);
  ++column_;
  return invalidToken;
}

void Lexer::OpenFile(const std::string &filename) {
  sourceFileStream_.open(filename);
  if ((sourceFileStream_.rdstate() & std::ifstream::failbit) != 0) {
    CompileMessage(filename, "warning: fail to open file.");
    exit(EXIT_FAILURE);
  }
}

void Lexer::ReadLine() {
  column_ = 0;
  if (!filename_.empty() && sourceFileStream_.is_open()) {
    std::getline(sourceFileStream_, line_);
    LOG_OUT << "-------------line-------------: \"" << line_ << "\"";
    if ((sourceFileStream_.rdstate() & std::ifstream::eofbit) != 0) {
      LOG_OUT << "Reach end of file for " << filename_;
      eof_ = true;
    } else if ((sourceFileStream_.rdstate() & std::ifstream::failbit) != 0) {
      CompileMessage(filename_, lineno_, 0, "warning: fail to read line.");
      exit(EXIT_FAILURE);
    }
  } else {
    std::getline(sourceStringStream_, line_, '\n');
    LOG_OUT << "-------------line-------------: \"" << line_ << "\"";
    if ((sourceStringStream_.rdstate() & std::ifstream::eofbit) != 0) {
      LOG_OUT << "Reach end of string lines";
      eof_ = true;
    } else if ((sourceStringStream_.rdstate() & std::ifstream::failbit) != 0) {
      CompileMessage(filename_, lineno_, 0, "warning: fail to read line.");
      exit(EXIT_FAILURE);
    }
  }
  ++lineno_;
}

bool Lexer::IsLineEnd() const {
  return line_.empty() || column_ == line_.length();
}

void Lexer::SetLineInfo(TokenPtr token) {
  token->lineStart = token->lineEnd = lineno_;
  token->columnStart = column_;
  token->columnEnd = column_ + token->name.size();
}

Token Lexer::GetOperator() {
  Token tok = TraverseOpTable(line_.c_str() + column_);
  SetLineInfo(&tok);
  return tok;
}

Token Lexer::GetSeparator() {
  Token tok = TraverseSpTable(line_.c_str() + column_);
  SetLineInfo(&tok);
  return tok;
}

Token Lexer::GetKeyword() {
  Token tok = TraverseKwTable(line_.c_str() + column_);
  SetLineInfo(&tok);
  return tok;
}

Token Lexer::GetLiteral() {
  Token tok = FindLiteral(line_.c_str() + column_);
  SetLineInfo(&tok);
  if (tok.type == TokenType_InvalidString) {
    // Exception here.
    CompileMessage(filename_, lineno_, column_,
                   "warning: unexcepted literal string format.");
    exit(EXIT_FAILURE);
  }
  if (tok.type != TokenType_ContinuousString) {
    return tok;
  }
  // String across multiple lines.
  int lineno = lineno_;
  int column = column_;
  for (EVER) {
    if (eof_) {
      // Exception here.
      CompileMessage(filename_, lineno, column,
                     "warning: unexcepted end of file during "
                     "scanning multiple lines string.");
      exit(EXIT_FAILURE);
    }
    ReadLine();
    const char *column = strchr(line_.c_str(), *tok.data.str);
    if (column != nullptr) { // Found the end of string.
      tok.name.append(1, '\n');
      auto len = column - line_.c_str(); // No ' or ".
      tok.name.append(line_.c_str(), len);
      tok.type = TokenType_Literal;
      tok.data.lt = LiteralId_str;
      tok.lineEnd = lineno_;
      tok.columnEnd = column - line_.c_str();
      column_ += len + 1; // With ' or ".
      return tok;
    } else {
      tok.name.append(1, '\n');
      tok.name.append(line_.c_str());
    }
  }
  return tok;
}

Token Lexer::GetIdentifier() {
  Token tok = FindIdentifier(line_.c_str() + column_);
  SetLineInfo(&tok);
  return tok;
}

Token Lexer::GetComment() {
  Token tok = FindComment(line_.c_str() + column_, line_.length() - column_);
  SetLineInfo(&tok);
  return tok;
}

std::string Lexer::UnescapeString(const std::string &str) {
  if (str.find('\\') == std::string::npos) {
    return str;
  }
  std::stringstream ss;
  std::string::const_iterator it = str.cbegin();
  while (it != str.cend()) {
    char c = *it++;
    if (c == '\\') {
      if (it == str.cend()) {
        // Invalid or unsupported escape sequence.
        auto invalidPos =
            std::distance(str.cbegin(), it - 1); // The position of /
        constexpr auto quotSize = 1;
        auto colPos = column_ - (str.size() - invalidPos) - quotSize;
        std::stringstream ss;
        ss << "error: unexpected string literal: '" << str
           << "', position: " << invalidPos << ", col: " << colPos;
        CompileMessage(filename_, lineno_, colPos, ss.str());
        exit(EXIT_FAILURE);
      }
      // https://en.cppreference.com/w/cpp/language/escape
      switch (*it++) {
      case '\'':
        c = '\'';
        break;
      case '"':
        c = '\"';
        break;
      case '?':
        c = '\?';
        break;
      case '\\':
        c = '\\';
        break;
      case 'a':
        c = '\a';
        break;
      case 'b':
        c = '\b';
        break;
      case 'f':
        c = '\f';
        break;
      case 'n':
        c = '\n';
        break;
      case 'r':
        c = '\r';
        break;
      case 't':
        c = '\t';
        break;
      case 'v':
        c = '\v';
        break;
      default:
        // Invalid or unsupported escape sequence.
        auto invalidPos =
            std::distance(str.cbegin(), it - 2); // The position of /
        constexpr auto quotSize = 1;
        auto colPos = column_ - (str.size() - invalidPos) - quotSize;
        std::stringstream ss;
        ss << "error: unexpected string literal: '" << str
           << "', position: " << invalidPos << ", col: " << colPos;
        CompileMessage(filename_, lineno_, colPos, ss.str());
        exit(EXIT_FAILURE);
      }
    }
    ss << c;
  }
  return ss.str();
}

std::string Lexer::EscapeString(const std::string &str) {
  return ConvertEscapeString(str);
}

void Lexer::Dump() {
  std::cout << "--------------------" << std::endl;
  std::cout << "------ token -------" << std::endl;
  for (const auto &token : Tokens()) {
    if (token.type == TokenType_End) {
      LOG_OUT << "No token anymore";
      break;
    }
    if (token.IsSeparatorSpace()) {
      continue;
    }
    std::string escapeName = EscapeString(token.name);
    std::cout << std::setfill(' ') << std::setw(30) << std::left << escapeName
              << "[" << ToStr(&token) << "]" << std::endl;
  }
}
} // namespace lexer
} // namespace da