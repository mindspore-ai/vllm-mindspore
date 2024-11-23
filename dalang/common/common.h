#ifndef __COMMON_H__
#define __COMMON_H__

#include <cstring>
#include <iostream>

#define LOG_OUT std::cout
#define LOG_ERROR std::cerr
#define LOG_ENDL std::endl

#define EVER                                                                   \
  ;                                                                            \
  ;

static inline void CompileMessage(const std::string &filename, const int line,
                                  const int col, const std::string &msg) {
  LOG_ERROR << filename << ':' << line << ':' << (col + 1) << ": " << msg
            << LOG_ENDL;
}

static inline void CompileMessage(const std::string &line_info,
                                  const std::string &msg) {
  LOG_ERROR << line_info << ": " << msg << LOG_ENDL;
}

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
#endif // __COMMON_H__