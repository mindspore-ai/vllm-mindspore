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

#define MACRO_HELPER(x) #x
#define MACRO(x) #x "=" MACRO_HELPER(x)
#pragma message(MACRO(__GNUC__))

#include "common/common.h"
#include "lang/c/compiler/compiler.h"
#include "lang/c/lexer/lexer.h"
#include "lang/cli/options.h"
#include "lang/c/parser/parser.h"
#include "lang/c/vm/vm.h"

#if __has_include(<filesystem>)
#pragma message("Using filesystem library: <filesystem>")
#include <filesystem>
namespace fs = std::filesystem;
#else
#pragma message("No filesystem library found.")
#define NO_FILESYSTEM_LIBRARY
namespace fs {
std::string Canonical(const std::string &path) {
#ifdef _MSC_VER
#pragma message("Not make canonical path.")
  return path;
#else
  char pathBuffer[PATH_MAX + 1] = {0};
  return std::string(realpath(path.c_str(), pathBuffer));
#endif
}
}  // namespace fs
#endif

using namespace da;

/* Command-Line Interface */
int main(int argc, char **argv) {
  struct arguments args = GetOptions(argc, argv);

  LOG_OUT << "args: " << args.args[0] << ", " << args.lex << ", " << args.parse << ", " << args.compile << ", "
          << args.silent << ", " << args.interpret << ", " << args.output;

  // Handle the source file path.
  const std::string filenameArg = args.args[0];
#ifdef NO_FILESYSTEM_LIBRARY
  std::string filename = fs::Canonical(filenameArg);
#else
  const auto path = fs::path(filenameArg);
  std::string filename;
  try {
    filename = fs::canonical(path).string();
  } catch (fs::filesystem_error &ex) {
    std::cout << "error: wrong path: " << path << std::endl;
    exit(EXIT_FAILURE);
  }
#endif
  auto lexer = lexer::Lexer(filename);
  if (args.lex && !args.silent) {
    lexer.Dump();
  }

  auto parser = parser::Parser(&lexer);
  parser.ParseCode();
  if (args.parse && !args.silent) {
    parser.DumpAst();
  }

  auto compiler = compiler::Compiler(&parser);
  compiler.Compile();
  if (args.compile && !args.silent) {
    compiler.Dump();
  }

  if (args.interpret) {
    auto vm = vm::VM(&compiler);
    vm.Run();
  }
  return 0;
}
