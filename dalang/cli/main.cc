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

#include <filesystem>

#include "common/common.h"
#include "compiler/compiler.h"
#include "lexer/lexer.h"
#include "options.h"
#include "parser/parser.h"
#include "vm/vm.h"

#undef DEBUG
#ifndef DEBUG
#undef LOG_OUT
#define LOG_OUT NO_LOG_OUT
#endif

/* Command-Line Interface */
int main(int argc, char **argv) {
  struct arguments args = GetOptions(argc, argv);

  LOG_OUT << "args: " << args.args[0] << ", " << args.lex << ", " << args.parse
          << ", " << args.compile << ", " << args.silent << ", "
          << args.interpret << ", " << args.output;

  // Handle the source file path.
  const std::string filenameArg = args.args[0];
  const auto path = std::filesystem::path(filenameArg);
  std::string filename;
  try {
    filename = std::filesystem::canonical(path).string();
  } catch (std::filesystem::filesystem_error &ex) {
    std::cout << "error: wrong path: " << path << std::endl;
    exit(EXIT_FAILURE);
  }

  auto lexer = lexer::Lexer(filename);
  if (args.lex && !args.silent) {
    lexer.Dump();
    std::cout << std::endl;
  }

  auto parser = parser::Parser(&lexer);
  parser.ParseCode();
  if (args.parse && !args.silent) {
    parser.DumpAst();
    std::cout << std::endl;
  }

  auto compiler = compiler::Compiler(&parser);
  compiler.Compile();
  if (args.compile && !args.silent) {
    compiler.Dump();
    std::cout << std::endl;
  }

  if (args.interpret) {
    auto vm = vm::VM(&compiler);
    vm.Run();
  }
  return 0;
}
