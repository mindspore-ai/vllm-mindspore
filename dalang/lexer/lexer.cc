#include "lexer.h"

#include <filesystem>
#include <fstream>

namespace lexer {
Lexer::Lexer(const std::string &filename)
    : filename_{std::filesystem::canonical(std::filesystem::path(filename))
                    .string()},
      file_{nullptr}, eof_{false} {
  OpenFile(filename);
}

Lexer::~Lexer() {
  if (file_.is_open()) {
    file_.close();
  }
}

const Token Lexer::NextToken() {
  // Skip blank line.
  while (IsLineEnd()) {
    if (eof_) {
      return Token{.type = TokenType_End};
    }
    ReadLine();
  }
  return TokenInLine();
}

const std::vector<Token> &Lexer::Tokens() {
  if (scanned_) {
    return tokens_;
  }
  while (true) {
    auto token = NextToken();
    if (token.type == TokenType_End) {
#ifdef DEBUG
      LOG_OUT << "No token anymore" << LOG_ENDL;
#endif
      break;
    }
    if (token.IsSeparatorSpace()) {
      continue;
    }
#ifdef DEBUG
    LOG_OUT << "# token: " << token.name << "\t\t\t[" << ToStr(&token) << "]"
            << LOG_ENDL;
#endif
    tokens_.emplace_back(std::move(token));
  }
  scanned_ = true;
  return tokens_;
}

const Token Lexer::TokenInLine() {
  if (skipWhiteSpace_) {
    pos_ += SkipWhiteSpace(line_.c_str() + pos_);
    if (IsLineEnd()) {
      return Token{.type = TokenType_End};
    }
  }

  Token token = GetComment();
  if (token.type != TokenType_End) {
    pos_ += token.name.size();
    return token;
  }
  token = GetOperator();
  if (token.type != TokenType_End) {
    pos_ += token.name.size();
    return token;
  }
  token = GetSeparator();
  if (token.type != TokenType_End) {
    pos_ += token.name.size();
    return token;
  }
  token = GetKeyword();
  if (token.type != TokenType_End) {
    pos_ += token.name.size();
    return token;
  }
  token = GetLiteral();
  if (token.type != TokenType_End) {
    if (token.lineStart == token.lineEnd) { // Not multiple lines.
      pos_ += token.name.size();
    }
    return token;
  }
  token = GetIdentifier();
  if (token.type != TokenType_End) {
    pos_ += token.name.size();
    return token;
  }

  // Not match any excepted token, return a invalid token and move on.
  Token invalidToken = Token{.type = TokenType_Invalid};
  if (!IsLineEnd()) {
    invalidToken.name = line_.at(pos_);
  }
  SetLineInfo(&invalidToken);
  ++pos_;
  return invalidToken;
}

void Lexer::OpenFile(const std::string &filename) {
  file_.open(filename);
  if ((file_.rdstate() & std::ifstream::failbit) != 0) {
    CompileMessage(filename, "warning: fail to open file.");
    exit(-1);
  }
}

const std::string &Lexer::ReadLine() {
  pos_ = 0;
  std::getline(file_, line_);
#ifdef DEBUG
  LOG_OUT << "-------------line-------------: \"" << line_ << "\"" << LOG_ENDL;
#endif
  if ((file_.rdstate() & std::ifstream::eofbit) != 0) {
#ifdef DEBUG
    LOG_OUT << "Reach end of file for " << filename_ << LOG_ENDL;
#endif
    eof_ = true;
  } else if ((file_.rdstate() & std::ifstream::failbit) != 0) {
    CompileMessage(filename_, lineno_, 0, "warning: fail to read line.");
    exit(-1);
  }
  ++lineno_;
  return line_;
}

bool Lexer::IsLineEnd() const {
  return line_.empty() || pos_ == line_.length();
}

void Lexer::SetLineInfo(TokenPtr token) {
  token->lineStart = token->lineEnd = lineno_;
  token->columnStart = pos_;
  token->columnEnd = pos_ + token->name.size();
}

Token Lexer::GetOperator() {
  Token tok = TraverseOpTable(line_.c_str() + pos_);
  SetLineInfo(&tok);
  return tok;
}

Token Lexer::GetSeparator() {
  Token tok = TraverseSpTable(line_.c_str() + pos_);
  SetLineInfo(&tok);
  return tok;
}

Token Lexer::GetKeyword() {
  Token tok = TraverseKwTable(line_.c_str() + pos_);
  SetLineInfo(&tok);
  return tok;
}

Token Lexer::GetLiteral() {
  Token tok = FindLiteral(line_.c_str() + pos_);
  SetLineInfo(&tok);
  if (tok.type != TokenType_ContinuousString) {
    return tok;
  }
  // String across multiple lines.
  int lineno = lineno_;
  int pos = pos_;
  while (true) {
    if (eof_) {
      // Exception here.
      CompileMessage(filename_, lineno, pos,
                     "warning: unexcepted end of file during "
                     "scanning multiple lines string.");
      exit(1);
    }
    ReadLine();
    const char *pos = strchr(line_.c_str(), *tok.data.str);
    if (pos != nullptr) { // Found the end of string.
      tok.name.append(1, '\n');
      auto len = pos - line_.c_str() + 1;
      tok.name.append(line_.c_str(), len);
      tok.type = TokenType_Literal;
      tok.data.lt = LiteralId_str;
      tok.lineEnd = lineno_;
      tok.columnEnd = pos - line_.c_str();
      pos_ += len;
      return tok;
    } else {
      tok.name.append(1, '\n');
      tok.name.append(line_.c_str());
    }
  }
  return tok;
}

Token Lexer::GetIdentifier() {
  Token tok = FindIdentifier(line_.c_str() + pos_);
  SetLineInfo(&tok);
  return tok;
}

Token Lexer::GetComment() {
  Token tok = FindComment(line_.c_str() + pos_, line_.length() - pos_);
  SetLineInfo(&tok);
  return tok;
}
} // namespace lexer