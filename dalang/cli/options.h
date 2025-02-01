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

#ifndef __CLI_OPTIONS_H__
#define __CLI_OPTIONS_H__

#include <argp.h>

/* Used by main to communicate with parse_opt. */
struct arguments {
  const char *args[1];
  bool lex;
  bool parse;
  bool compile;
  bool silent;
  bool interpret;
  const char *output;
};

struct arguments GetOptions(int argc, char **argv);

#endif // __CLI_OPTIONS_H__