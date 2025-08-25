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

#include <string>
#include <iomanip>
#include <iostream>
#include <sstream>

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

static inline std::string GetTime() {
  auto t = time(0);
  tm lt;
  localtime_r(&t, &lt);
  std::stringstream ss;
  ss << (lt.tm_year + 1900) << '-' << (lt.tm_mon + 1) << '-' << std::setfill('0') << std::setw(2) << lt.tm_mday << ' '
     << std::setw(2) << lt.tm_hour << ':' << std::setw(2) << lt.tm_min << ':' << std::setw(2) << lt.tm_sec;
  return ss.str();
}

#define LOG_OUT Cout() << GetTime() << " [" << __FILE__ << ':' << __LINE__ << ' ' << __FUNCTION__ << "] "

#define LOG_ERROR Cerr() << GetTime() << " [" << __FILE__ << ':' << __LINE__ << ' ' << __FUNCTION__ << "] error: "

#define NO_LOG_OUT \
  while (false) Cout()

#define NO_LOG_ERROR \
  while (false) Cerr()

#ifndef DEBUG_LOG_OUT
#undef LOG_OUT
#define LOG_OUT NO_LOG_OUT
#endif

#endif  // __COMMON_LOGGER_H__
