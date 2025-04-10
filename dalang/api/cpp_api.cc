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
#include <cstring>
#include <iostream>

#include "compiler/compiler.h"
#include "lexer/lexer.h"
#include "parser/parser.h"
#include "vm/vm.h"

#undef DEBUG
#ifndef DEBUG
#undef LOG_OUT
#define LOG_OUT NO_LOG_OUT
#endif

using Callable = compiler::Compiler;

Callable *Compile(const char *source) {
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
#ifdef DEBUG
  lexer.Dump();
#endif

  auto parser = parser::Parser(&lexer);
  parser.ParseCode();
#ifdef DEBUG
  parser.DumpAst();
#endif

  auto compiler = new compiler::Compiler(&parser); // delete by user.
  compiler->Compile();
#ifdef DEBUG
  compiler->Dump();
#endif
  LOG_OUT << "return callable: " << compiler;
  return compiler;
}

void Run(tensor::DAGraph *callable, std::vector<tensor::DATensor *> inputs) {
  CHECK_NULL(callable);
  LOG_OUT << "run callable: " << callable;
  auto vm = vm::VM(callable);
  vm.Run();
}
