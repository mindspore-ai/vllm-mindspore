#include "lexer.h"

#include <fstream>

#undef DEBUG

namespace lexer {
Lexer::Lexer(const std::string &filename)
    : filename_{filename}, file_{nullptr}, eof_{false} {
  OpenFile(filename);
}

Lexer::~Lexer() {
  if (file_.is_open()) {
    file_.close();
  }
}

const Token Lexer::NextToken() {
  while (IsEnd()) { // Skip blank line.
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
    if (IsEnd()) {
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
    pos_ += token.name.size();
    return token;
  }
  token = GetIdentifier();
  if (token.type != TokenType_End) {
    pos_ += token.name.size();
    return token;
  }
  return Token{.type = TokenType_End};
}

void Lexer::OpenFile(const std::string &filename) {
  file_.open(filename);
  if ((file_.rdstate() & std::ifstream::failbit) != 0) {
    LOG_ERROR << "Fail to open " << filename << LOG_ENDL;
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
    LOG_ERROR << "Fail to read line for " << filename_ << LOG_ENDL;
    exit(-1);
  }
  ++lineno_;
  return line_;
}

bool Lexer::IsEnd() const { return line_.empty() || pos_ == line_.length(); }

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