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

#ifndef __LEXER_OPERATOR_H__
#define __LEXER_OPERATOR_H__

#define OPERATOR(O) OpId_##O,
typedef enum OperatorId {
#include "lexer/operator.list"
  OpId_End,
} OpId;
#undef OPERATOR

namespace lexer {
typedef struct NameToOperatorId {
  const char *name;
  OpId id;
} NameToOpId;

const char *ToStr(OpId opid);
} // namespace lexer

#endif // __LEXER_OPERATOR_H__