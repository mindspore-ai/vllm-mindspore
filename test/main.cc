#include "compiler/compiler.h"
#include "lexer/lexer.h"
#include "parser/parser.h"
#include "vm/vm.h"

#undef LOG_OUT
#define LOG_OUT LOG_NO_OUT

void RunLexerTest(const char *filename) {
  auto lexer = lexer::Lexer(filename);
  for (EVER) {
    auto token = lexer.NextToken();
    if (token.type == lexer::TokenType_End) {
      LOG_OUT << "No token anymore";
      break;
    }
    if (token.IsSeparatorSpace()) {
      continue;
    }
    LOG_OUT << "# token: " << token.name << "\t\t\t[" << ToStr(&token) << "]";
  }
}

void RunParserTest(const char *filename) {
  auto parser = parser::Parser(filename);
  parser.ParseCode();
  parser.DumpAst();
}

void RunCompilerTest(const char *filename) {
  auto compiler = compiler::Compiler(filename);
  compiler.Compile();
  compiler.Dump();
}

void RunCompilerAndVmTest(const char *filename) {
  auto compiler = compiler::Compiler(filename);
  compiler.Compile();
  compiler.Dump();
  auto vm = vm::VM(&compiler);
  vm.Run();
}

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "Need a file name" << std::endl;
    return -1;
  }
  auto filename = argv[1];
  LOG_OUT << "\nfilename: " << filename;
  constexpr auto test_lexer = false;
  if (test_lexer) {
    RunLexerTest(filename);
  }
  constexpr auto test_parser = false;
  if (test_parser) {
    RunParserTest(filename);
  }
  constexpr auto test_compiler = false;
  if (test_compiler) {
    RunCompilerTest(filename);
  }
  constexpr auto test_vm = true;
  if (test_vm) {
    RunCompilerAndVmTest(filename);
  }
  return 0;
}