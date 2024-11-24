#ifndef __COMMON_H__
#define __COMMON_H__

#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>

class Cout {
public:
  Cout() = default;
  ~Cout() { std::cout << std::endl; }

  template <typename T> Cout &operator<<(const T &val) noexcept {
    std::cout << val;
    return *this;
  }
};

class Cerr {
public:
  Cerr() = default;
  ~Cerr() { std::cerr << std::endl; }

  template <typename T> Cerr &operator<<(const T &val) noexcept {
    std::cerr << val;
    return *this;
  }
};

static inline std::string GetTime() {
  auto t = time(0);
  const tm *lt = localtime(&t);
  std::stringstream ss;
  ss << (lt->tm_year + 1900) << '-' << lt->tm_mon << '-' << std::setfill('0')
     << std::setw(2) << lt->tm_mday << ' ' << std::setw(2) << lt->tm_hour << ':'
     << std::setw(2) << lt->tm_min << ':' << std::setw(2) << lt->tm_sec;
  return ss.str();
}

#define LOG_OUT                                                                \
  Cout() << GetTime() << " [" << __FILE__ << ':' << __LINE__ << ' '            \
         << __FUNCTION__ << "] "
#define LOG_ERROR                                                              \
  Cerr() << GetTime() << " [" << __FILE__ << ':' << __LINE__ << ' '            \
         << __FUNCTION__ << "] error: "
#define ENDL std::endl

#define TO_STR(s) #s
#define CHECK_NULL(a)                                                          \
  if (a == nullptr) {                                                          \
    LOG_ERROR << '\'' << TO_STR(a) << "\' should not be null.";                \
    exit(1);                                                                   \
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