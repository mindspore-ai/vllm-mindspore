/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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
#include "lang/api/c_api.h"

#include <cstring>
#include <iostream>

#ifdef __cplusplus
extern "C" {
#endif
Callable *DA_API_Compile(const char *source, bool graph, bool dump) {
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

  auto lexer = da::lexer::Lexer(functionStr);
  if (dump) {
    lexer.Dump();
  }

  auto parser = da::parser::Parser(&lexer);
  parser.ParseCode();
  if (dump) {
    parser.DumpAst();
  }

  auto compiler = da::compiler::Compiler(&parser, true, graph);  // delete by user.
  compiler.Compile();
  if (dump) {
    compiler.Dump();
  }

  auto vm = new da::vm::VM(&compiler, true);
  LOG_OUT << "Return callable: " << vm;
  return vm;
}

Result DA_API_Run(Callable *callable, const std::vector<Argument> &args) {
  CHECK_IF_NULL(callable);
  LOG_OUT << "Run callable: " << callable;
  return callable->Run(args);
}
#ifdef __cplusplus
}
#endif
