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

#ifndef __API_C_API_H__
#define __API_C_API_H__

#include "compiler/compiler.h"
#include "lexer/lexer.h"
#include "parser/parser.h"
#include "tensor/da_tensor.h"
#include "vm/vm.h"

using Callable = compiler::Compiler;
using Argument = vm::Argument;
using Result = vm::Result;
using Tensor = tensor::DATensor;

#ifdef __cplusplus
extern "C" {
#endif
/// \brief Compile the source code with dalang compiler, and return a callable.
/// \param[in] source The source code string.
/// \param[in] dump If dump the compiler information.
/// \return A callable object, which should be freed by user outside.
Callable *DA_API_Compile(const char *source, bool dump);

/// \brief Run the callable object returned from 'DA_API_Compile()'.
/// \param[in] callable The callable object.
/// \param[in] args The arguments.
/// \return The result of running callable.
Result DA_API_Run(Callable *callable, const std::vector<Argument> &args);
#ifdef __cplusplus
}
#endif
#endif // __API_C_API_H__