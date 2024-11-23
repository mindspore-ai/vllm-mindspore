#include "lexer/lexer.h"
#include "parser/parser.h"

void RunLexerTest(const char *filename) {
  auto lexer = lexer::Lexer(filename);
  for (EVER) {
    auto token = lexer.NextToken();
    if (token.type == lexer::TokenType_End) {
      LOG_OUT << "No token anymore" << LOG_ENDL;
      break;
    }
    if (token.IsSeparatorSpace()) {
      continue;
    }
    LOG_OUT << "# token: " << token.name << "\t\t\t[" << ToStr(&token) << "]"
            << LOG_ENDL;
  }
}

void RunParserTest(const char *filename) {
  auto parser = parser::Parser(filename);
  parser.ParseCode();
  parser.DumpAst();
}

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "Need a file name" << std::endl;
    return -1;
  }
  auto filename = argv[1];
  LOG_OUT << "filename: " << filename << LOG_ENDL;
  constexpr auto test_lexer = false;
  if (test_lexer) {
    RunLexerTest(filename);
  }
  RunParserTest(filename);
  return 0;
}