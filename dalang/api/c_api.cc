/**
 * Copyright 2025 Zhang Qinghua
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
#include "c_api.h"

#include <cstring>
#include <iostream>

#undef DEBUG
#ifndef DEBUG
#undef LOG_OUT
#define LOG_OUT NO_LOG_OUT
#endif

#ifdef __cplusplus
extern "C" {
#endif
Callable *DA_API_Compile(const char *source,
                         const std::vector<const Tensor *> &args, bool dump) {
  if (source == nullptr) {
    LOG_ERROR << "error: no source code.";
    exit(EXIT_FAILURE);
  }
  LOG_OUT << "source:\n" << source;
  constexpr auto pythonDef = "def";
  const char *functionStr = strstr(source, pythonDef);
  if (functionStr == nullptr) {
    LOG_ERROR << "error: keyword 'def' not found.";
    exit(EXIT_FAILURE);
  }

  auto lexer = lexer::Lexer(functionStr);
  if (dump) {
    lexer.Dump();
  }

  auto parser = parser::Parser(&lexer);
  parser.ParseCode();
  if (dump) {
    parser.DumpAst();
  }

  auto compiler = new compiler::Compiler(&parser, true); // delete by user.
  compiler->Compile();
  if (dump) {
    compiler->Dump();
  }

  LOG_OUT << "Return callable: " << compiler;
  return compiler;
}

void DA_API_Run(Callable *callable, const std::vector<const Tensor *> &args) {
  CHECK_NULL(callable);
  LOG_OUT << "Run callable: " << callable;
  auto vm = vm::VM(callable);
  vm.Run();
}

#ifdef __cplusplus
}
#endif