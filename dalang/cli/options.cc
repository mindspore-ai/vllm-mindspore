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

#include "options.h"
#include "common/logger.h"

#include <cstring>
#include <iostream>

#undef DEBUG
#ifndef DEBUG
#undef LOG_OUT
#define LOG_OUT NO_LOG_OUT
#endif

/* ...snippet

We now use the first four fields in ARGP, so here's a description of them:
  OPTIONS  -- A pointer to a vector of struct argp_option (see below)
  PARSER   -- A function to parse a single option, called by argp
  ARGS_DOC -- A string describing how the non-option arguments should look
  DOC      -- A descriptive string about this program; if it contains a
              vertical tab character (\v), the part after it will be
              printed *following* the options

The function PARSER takes the following arguments:
  KEY  -- An integer specifying which option this is (taken
          from the KEY field in each struct argp_option), or
          a special key specifying something else; the only
          special keys we use here are ARGP_KEY_ARG, meaning
          a non-option argument, and ARGP_KEY_END, meaning
          that all arguments have been parsed
  ARG  -- For an option KEY, the string value of its
          argument, or NULL if it has none
  STATE-- A pointer to a struct argp_state, containing
          various useful information about the parsing state; used here
          are the INPUT field, which reflects the INPUT argument to
          argp_parse, and the ARG_NUM field, which is the number of the
          current non-option argument being parsed
It should return either 0, meaning success, ARGP_ERR_UNKNOWN, meaning the
given KEY wasn't recognized, or an errno value indicating some other
error.

Note that in this example, main uses a structure to communicate with the
parse_opt function, a pointer to which it passes in the INPUT argument to
argp_parse.  Of course, it's also possible to use global variables
instead, but this is somewhat more flexible.

The OPTIONS field contains a pointer to a vector of struct argp_option's;
that structure has the following fields (if you assign your option
structures using array initialization like this example, unspecified
fields will be defaulted to 0, and need not be specified):
  NAME   -- The name of this option's long option (may be zero)
  KEY    -- The KEY to pass to the PARSER function when parsing this option,
            *and* the name of this option's short option, if it is a
            printable ascii character
  ARG    -- The name of this option's argument, if any
  FLAGS  -- Flags describing this option; some of them are:
              OPTION_ARG_OPTIONAL -- The argument to this option is optional
              OPTION_ALIAS        -- This option is an alias for the
                                      previous option
              OPTION_HIDDEN       -- Don't show this option in --help output
  DOC    -- A documentation string for this option, shown in --help output

An options vector should be terminated by an option with all fields zero.*/

const char *argp_program_version = "da-lang v0.1";
const char *argp_program_bug_address = "<zhangqinghua3@huawei.com>";

/* Program documentation. */
static char doc[] = "da -- a program to compile and run da-lang codes.";

/* A description of the arguments we accept. */
static char args_doc[] = "*.da"; /* "*.da *.da ..." */

/* The options we provide. */
static struct argp_option options[] = {
    {"verbose", 'v', 0, 0,
     "Print verbose output, include tokens, AST and bytecode\nOption combined "
     "-l -p -c"},
    {"silent", 's', 0, 0,
     "Don't print any compile output\nOption opposite to -v"},
    {"lex", 'l', 0, 0, "Print the tokens output"},
    {"parse", 'p', 0, 0, "Print the AST output"},
    {"compile", 'c', 0, 0, "Print the bytecode output"},
    {"run", 'r', "ENABLED", 0,
     "Interpret the code if ENABLED is not 0 or \"disable\"\nEnabled by "
     "default"},
    {"interpret", 0, 0, OPTION_ALIAS},
    {"output", 'o', "FILE", 0,
     "Output the bytecode as FILE for later execution\n[to-be-supported]"},
    {0}};

/* Parse a single option. */
static error_t ParseOption(int key, char *arg, struct argp_state *state) {
  /* Get the input argument from argp_parse, which we
     know is a pointer to our arguments structure. */
  struct arguments *arguments = (struct arguments *)state->input;
  LOG_OUT << "key: " << std::hex << key;
  switch (key) {
  case 'v':
    arguments->lex = true;
    arguments->parse = true;
    arguments->compile = true;
    break;

  case 's':
    arguments->silent = true;
    break;

  case 'l':
    arguments->lex = true;
    break;
  case 'p':
    arguments->parse = true;
    break;
  case 'c':
    arguments->compile = true;
    break;
  case 'r':
    arguments->interpret =
        strcmp(arg, "disable") == 0 || strcmp(arg, "0") == 0 ? false : true;
    break;
  case 'o':
    arguments->output = arg;
    break;

  case ARGP_KEY_INIT:
    break;

  case ARGP_KEY_FINI:
    break;

  case ARGP_KEY_ARG:
    LOG_OUT << "state->arg_num: " << state->arg_num;
    if (state->arg_num >= 1) {
      /* Too many arguments. */
      argp_usage(state);
    }
    arguments->args[state->arg_num] = arg;
    break;

  case ARGP_KEY_END:
    if (state->arg_num < 1) {
      /* Not enough arguments. */
      argp_usage(state);
    }
    break;

  default:
    return ARGP_ERR_UNKNOWN;
  }
  LOG_OUT << "args: " << arguments->args[0] << ", " << arguments->lex << ", "
          << arguments->parse << ", " << arguments->compile << ", "
          << arguments->interpret << ", " << arguments->output;
  return 0;
}

/* Our argp parser. */
static struct argp argp = {options, ParseOption, args_doc, doc};

struct arguments GetOptions(int argc, char **argv) {
  struct arguments arguments = {0};

  /* Default values. */
  arguments.args[0] = "?";
  arguments.lex = false;
  arguments.parse = false;
  arguments.compile = false;
  arguments.silent = false;
  arguments.interpret = true;
  arguments.output = "?";

  argp_parse(&argp, argc, argv, 0, 0, &arguments);
  return arguments;
}