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

#ifndef __COMMON_COMMON_H__
#define __COMMON_COMMON_H__

#include <codecvt>
#include <cstring>
#include <iostream>
#include <locale>
#include <sstream>
#include <string>

#define ENDL '\n'

#define TO_STR(s) #s
#define CHECK_NULL(a)                                                          \
  if (a == nullptr) {                                                          \
    LOG_ERROR << '\'' << TO_STR(a) << "\' should not be null.";                \
    exit(EXIT_FAILURE);                                                        \
  }

#define EVER                                                                   \
  ;                                                                            \
  ;

static inline void CompileMessage(const std::string &filename, const int line,
                                  const int col, const std::string &msg) {
  std::cout << filename << ':' << line << ':' << (col + 1) << ": " << msg
            << '\n';
}

static inline void CompileMessage(const std::string &line_info,
                                  const std::string &msg) {
  std::cout << line_info << ": " << msg << '\n';
}

// Skip blank, return blank count.
static inline size_t SkipWhiteSpace(const char *str) {
  size_t pos = 0;
  while (str[pos] != '\0') {
    char c = str[pos];
    switch (c) {
    case ' ':
    case '\r':
    case '\t':
      ++pos;
      break;
    default:
      return pos;
    }
  }
  return pos;
}

template <typename T>
int FindNameIndex(const char *str, T *table, size_t tableSize) {
  const auto strLen = strlen(str);
  for (size_t i = 0; i < tableSize; ++i) {
    T &element = table[i];
    const auto elementNameLen = strlen(element.name);
    if (elementNameLen <= strLen &&
        strncmp(str, element.name, elementNameLen) == 0) {
      return i;
    }
  }
  return -1;
}

inline std::wstring StringToWString(const std::string &str) {
  std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
  return converter.from_bytes(str);
}

inline std::string WStringToString(const std::wstring &wstr) {
  std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
  return converter.to_bytes(wstr);
}

inline std::string ConvertEscapeString(const std::string &str) {
  std::stringstream ss;
  std::string::const_iterator it = str.cbegin();
  while (it != str.cend()) {
    char c = *it++;
    // https://en.cppreference.com/w/cpp/language/escape
    switch (c) {
    case '\'':
      ss << "\'";
      break;
    case '\"':
      ss << "\\\"";
      break;
    case '\?':
      ss << "\\?";
      break;
    case '\\':
      ss << "\\\\";
      break;
    case '\a':
      ss << "\\a";
      break;
    case '\b':
      ss << "\\b";
      break;
    case '\f':
      ss << "\\f";
      break;
    case '\n':
      ss << "\\n";
      break;
    case '\r':
      ss << "\\r";
      break;
    case '\t':
      ss << "\\t";
      break;
    case '\v':
      ss << "\\v";
      break;
    default:
      ss << c;
      break;
    }
  }
  return ss.str();
}
#endif // __COMMON_COMMON_H__