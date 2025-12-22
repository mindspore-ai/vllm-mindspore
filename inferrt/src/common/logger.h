/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#ifndef __COMMON_LOGGER_H__
#define __COMMON_LOGGER_H__

#include <unistd.h>
#include <thread>
#include <string>
#include <iomanip>
#include <vector>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <ctime>

static inline std::string GetTime() {
  auto t = time(0);
  tm lt;
  localtime_r(&t, &lt);
  std::stringstream ss;
  ss << (lt.tm_year + 1900) << '-' << (lt.tm_mon + 1) << '-' << std::setfill('0') << std::setw(2) << lt.tm_mday << ' '
     << std::setw(2) << lt.tm_hour << ':' << std::setw(2) << lt.tm_min << ':' << std::setw(2) << lt.tm_sec;
  return ss.str();
}

template <typename T>
std::ostream &operator<<(std::ostream &os, const std::vector<T> &vec) noexcept {
  os << "{";
  for (size_t i = 0; i < vec.size(); ++i) {
    if (i > 0) {
      os << ", ";
    }
    os << std::to_string(vec[i]);
  }
  os << "}";
  return os;
}

class Cout {
 public:
  Cout() = default;
  ~Cout() { std::cout << std::endl; }

  template <typename T>
  Cout &operator<<(const T &val) noexcept {
    std::cout << val;
    return *this;
  }
};

class Cerr {
 public:
  Cerr() = default;
  ~Cerr() { std::cerr << std::endl; }

  template <typename T>
  Cerr &operator<<(const T &val) noexcept {
    std::cerr << val;
    return *this;
  }
};

class Cexception {
 public:
  Cexception(const char *file, int line, const char *func) {
    prefix_ << GetTime() << " [(pid:" << getpid() << ", thread id:" << std::hex << std::this_thread::get_id()
            << std::dec << ") " << file << ':' << line << ' ' << func << "] exception: ";
  }
  ~Cexception() noexcept(false) {
    std::string msg = msg_.str();
    std::string prefix = prefix_.str();
    std::cerr << prefix << msg << std::endl;
    throw std::runtime_error(msg);
  }

  template <typename T>
  Cexception &operator<<(const T &val) {
    msg_ << val;
    return *this;
  }

 private:
  std::stringstream prefix_;
  std::stringstream msg_;
};

#define LOG_OUT                                                                                                        \
  Cout() << GetTime() << " [(pid:" << getpid() << ", thread id:" << std::hex << std::this_thread::get_id() << std::dec \
         << ") " << __FILE__ << ':' << __LINE__ << ' ' << __FUNCTION__ << "] "

#define LOG_ERROR                                                                                                      \
  Cerr() << GetTime() << " [(pid:" << getpid() << ", thread id:" << std::hex << std::this_thread::get_id() << std::dec \
         << ") " << __FILE__ << ':' << __LINE__ << ' ' << __FUNCTION__ << "] error: "

namespace mrt {
namespace common {

// Stream-style exception builder that keeps compiler-visible [[noreturn]] semantics.
//
// Motivation:
// - Legacy LOG_EXCEPTION throws from Cexception destructor, which is not visible
//   as "noreturn" to the compiler and can trigger "missing return" diagnostics
//   in non-void functions under -Werror.
//
// Usage:
//   LOG_EXCEPTION << "bad value: " << v;
//
// NOTE: We intentionally pass (file, line, func) from the call-site so the
// thrown exception points to the real error location rather than this helper.
class ThrowStream final {
 public:
  ThrowStream(const char *file, int line, const char *func) : file_(file), line_(line), func_(func) {}

  std::ostringstream &Stream() { return oss_; }

  [[noreturn]] void ThrowNow() {
    ::Cexception(file_, line_, func_) << oss_.str();
    __builtin_unreachable();
  }

 private:
  const char *file_;
  int line_;
  const char *func_;
  std::ostringstream oss_;
};

// A null sink used to keep NO_LOG_EXCEPTION well-formed and streamable.
class NullStream final {
 public:
  template <typename T>
  NullStream &operator<<(const T &) noexcept {
    return *this;
  }
};

}  // namespace common
}  // namespace mrt

#define MRT_LOG_CONCAT_INNER_(a, b) a##b
#define MRT_LOG_CONCAT_(a, b) MRT_LOG_CONCAT_INNER_(a, b)

// Implemented as a for-loop:
// - the body performs stream insertion
// - the increment expression calls a [[noreturn]] function that throws
#define LOG_EXCEPTION_IMPL_(id)                                                                                       \
  for (::mrt::common::ThrowStream MRT_LOG_CONCAT_(_mrt_log_exception_stream_, id)(__FILE__, __LINE__, __FUNCTION__);; \
       MRT_LOG_CONCAT_(_mrt_log_exception_stream_, id).ThrowNow())                                                    \
  MRT_LOG_CONCAT_(_mrt_log_exception_stream_, id).Stream()

#define LOG_EXCEPTION LOG_EXCEPTION_IMPL_(__COUNTER__)

#define NO_LOG_OUT \
  while (false) Cout()

#define NO_LOG_ERROR \
  while (false) Cerr()

#define NO_LOG_EXCEPTION \
  while (false) ::mrt::common::NullStream()

#ifndef DEBUG_LOG_OUT
#undef LOG_OUT
#define LOG_OUT NO_LOG_OUT
#endif

#endif  // __COMMON_LOGGER_H__
