/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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

#include "common/logger.h"

#include <cerrno>
#include <cctype>
#include <climits>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>

#ifdef USE_GLOG
#ifdef LOG
#undef LOG
#endif
#ifdef VLOG
#undef VLOG
#endif
#ifdef VLOG_IS_ON
#undef VLOG_IS_ON
#endif
#define GLOG_NO_ABBREVIATED_SEVERITIES
#define google mrt_private
#define GLOG_USE_GLOG_EXPORT
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "glog/logging.h"
#pragma GCC diagnostic pop
#undef google
#ifdef LOG
#undef LOG
#endif
#ifdef VLOG
#undef VLOG
#endif
#ifdef VLOG_IS_ON
#undef VLOG_IS_ON
#endif
#endif

#ifndef _MSC_VER
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>
#else
#include <direct.h>
#endif

#include "common/common.h"

namespace mrt {
namespace common {

uint64_t g_mrt_vlog_mask = 0;

namespace {
constexpr int kDefaultLogLevel = static_cast<int>(GLogLevel::WARNING);
constexpr char kSplitLine[] = "----------------------------------------------------\n";
constexpr int kVLogMaxLevel = static_cast<int>(kMaxVLogLevel);

struct LogConfig {
  int glog_level = kDefaultLogLevel;
};

struct VLogRangeDesc {
  VLogLevel begin;
  VLogLevel end;
  const char *module;
};

struct VLogTagDesc {
  VLogLevel level;
  const char *name;
  const char *description;
};

LogConfig &GetConfig() {
  static LogConfig config;
  return config;
}

std::once_flag &GetInitFlag() {
  static std::once_flag init_flag;
  return init_flag;
}

bool IsDigit(char ch) { return ch >= '0' && ch <= '9'; }

bool ParseLogLevel(const std::string &value, int *level) {
  if (value.size() != 1 || !IsDigit(value[0])) {
    return false;
  }
  int parsed_level = value[0] - '0';
  if (parsed_level < static_cast<int>(GLogLevel::DEBUG) || parsed_level > static_cast<int>(GLogLevel::CRITICAL)) {
    return false;
  }
  *level = parsed_level;
  return true;
}

bool ParseNonNegativeInt(const std::string &value, size_t begin, size_t end, int *result) {
  if (begin >= end) {
    return false;
  }

  int64_t parsed_value = 0;
  size_t index = begin;
  while (index < end && IsDigit(value[index])) {
    parsed_value = parsed_value * 10 + (value[index] - '0');
    if (parsed_value > std::numeric_limits<int>::max()) {
      return false;
    }
    ++index;
  }
  if (index != end) {
    return false;
  }
  *result = static_cast<int>(parsed_value);
  return true;
}

std::string GetTimeString() {
#if defined(_WIN32) || defined(_WIN64)
  return "";
#else
  constexpr size_t kBufLen = 80;
  constexpr int kWidth = 3;
  constexpr int64_t kUsecToMsec = 1000;
  char buf[kBufLen] = {'\0'};
  timeval cur_time;
  (void)gettimeofday(&cur_time, nullptr);
  tm now;
  (void)localtime_r(&cur_time.tv_sec, &now);
  (void)strftime(buf, kBufLen, "%Y-%m-%d-%H:%M:%S", &now);
  std::stringstream ss;
  ss << buf << "." << std::setfill('0') << std::setw(kWidth) << cur_time.tv_usec / kUsecToMsec << "."
     << std::setfill('0') << std::setw(kWidth) << cur_time.tv_usec % kUsecToMsec;
  return ss.str();
#endif
}

std::string Trim(const std::string &value) {
  size_t begin = 0;
  while (begin < value.size() && std::isspace(static_cast<unsigned char>(value[begin])) != 0) {
    ++begin;
  }
  size_t end = value.size();
  while (end > begin && std::isspace(static_cast<unsigned char>(value[end - 1])) != 0) {
    --end;
  }
  return value.substr(begin, end - begin);
}

bool ParseVLogLevel(const std::string &value, int *level) {
  const auto token = Trim(value);
  if (token.empty() || !ParseNonNegativeInt(token, 0, token.size(), level)) {
    return false;
  }
  return *level <= kVLogMaxLevel;
}

bool ParseVLogToken(const std::string &token, uint64_t *mask) {
  const auto trimmed = Trim(token);
  if (trimmed.empty()) {
    return false;
  }

  auto dash = trimmed.find('-');
  if (dash == std::string::npos) {
    int level = 0;
    if (!ParseVLogLevel(trimmed, &level)) {
      return false;
    }
    *mask |= (uint64_t{1} << static_cast<uint8_t>(level));
    return true;
  }

  if (dash == 0 || dash + 1 >= trimmed.size() || trimmed.find('-', dash + 1) != std::string::npos) {
    return false;
  }

  int from = 0;
  int to = 0;
  if (!ParseVLogLevel(trimmed.substr(0, dash), &from) || !ParseVLogLevel(trimmed.substr(dash + 1), &to) || from > to) {
    return false;
  }

  for (int level = from; level <= to; ++level) {
    *mask |= (uint64_t{1} << static_cast<uint8_t>(level));
  }
  return true;
}

bool ParseVLogMask(const std::string &value, uint64_t *mask) {
  if (value.empty()) {
    return true;
  }

  // VLOG_v accepts comma-separated levels and inclusive ranges, for example:
  //   VLOG_v="1,2-3,8-15"
  // Parsed levels are stored as bits in g_mrt_vlog_mask, making IsVlogOn a single shift-and-test.
  uint64_t parsed_mask = 0;
  size_t begin = 0;
  while (begin <= value.size()) {
    const auto comma = value.find(',', begin);
    const auto end = comma == std::string::npos ? value.size() : comma;
    if (!ParseVLogToken(value.substr(begin, end - begin), &parsed_mask)) {
      return false;
    }
    if (comma == std::string::npos) {
      break;
    }
    begin = comma + 1;
  }

  *mask = parsed_mask;
  return true;
}

bool IsExceptionLevel(GLogLevel level) { return static_cast<int>(level) >= static_cast<int>(GLogLevel::CRITICAL); }

std::string GetLogLevelName(GLogLevel level) {
  static const char *const level_names[] = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"};
  return level_names[static_cast<int>(level)];
}

#ifdef USE_GLOG
mrt_private::LogSeverity GetGlogLevel(GLogLevel level) {
  switch (level) {
    case GLogLevel::DEBUG:
    case GLogLevel::INFO:
      return mrt_private::GLOG_INFO;
    case GLogLevel::WARNING:
      return mrt_private::GLOG_WARNING;
    case GLogLevel::ERROR:
    case GLogLevel::CRITICAL:
    default:
      return mrt_private::GLOG_ERROR;
  }
}

int GetThresholdLevel(const std::string &threshold) {
  if (threshold.empty()) {
    return mrt_private::GLOG_WARNING;
  }
  if (threshold == "DEBUG" || threshold == "INFO") {
    return mrt_private::GLOG_INFO;
  }
  if (threshold == "WARNING") {
    return mrt_private::GLOG_WARNING;
  }
  if (threshold == "ERROR" || threshold == "CRITICAL") {
    return mrt_private::GLOG_ERROR;
  }
  return mrt_private::GLOG_WARNING;
}

bool MakeDirectory(const char *path) {
#if defined(_WIN32) || defined(_WIN64)
  if (mkdir(path) == -1) {
#else
  constexpr int kDefaultMkdirMode = 0700;
  if (mkdir(path, kDefaultMkdirMode) == -1) {
#endif
    if (errno != EEXIST) {
      return false;
    }
  }
  return true;
}

bool MakePath(char *file_path) {
  if (file_path == nullptr || file_path[0] == '\0') {
    return true;
  }

  char *p = file_path + 1;
  while (*p) {
    if (*p == '\\' || *p == '/') {
      char ch = *p;
      *p = '\0';
      if (!MakeDirectory(file_path)) {
        *p = ch;
        return false;
      }
      *p = ch;
    }
    ++p;
  }
  return MakeDirectory(file_path);
}

std::string GetRealLogPath(const std::string &log_dir) {
  const std::string rank_id = GetEnv("RANK_ID");
  const std::string mpi_rank_id = GetEnv("OMPI_COMM_WORLD_RANK");
  if (!rank_id.empty()) {
    return log_dir + "/rank_" + rank_id + "/logs";
  }
  if (!mpi_rank_id.empty()) {
    return log_dir + "/rank_" + mpi_rank_id + "/logs";
  }
  return log_dir + "/pid_" + std::to_string(getpid()) + "/logs";
}
#endif

void OutputWarningBeforeInit(const std::string &msg) { std::cerr << "[WARNING] MRT: " << msg << std::endl; }

void PrintVLogTag(const char *file, int line, const char *function, const std::string &msg) {
  LogWriter(file, line, function, VLogLevel::DISP_VLOG_TAGS) < LogStream() << msg;
}

void DispVLogTags() {
  if (!IsVlogOn(VLogLevel::DISP_VLOG_TAGS)) {
    return;
  }

  static constexpr VLogRangeDesc kVLogRangeDescs[] = {
    {VLogLevel::RUNTIME, VLogLevel::RUNTIME_LAST, "runtime"},
    {VLogLevel::OPS, VLogLevel::OPS_LAST, "ops"},
    {VLogLevel::HARDWARE, VLogLevel::HARDWARE_LAST, "hardware"},
    {VLogLevel::COMMON, VLogLevel::COMMON_LAST, "common"},
    {VLogLevel::CONFIG, VLogLevel::CONFIG_LAST, "config"},
    {VLogLevel::IR, VLogLevel::IR_LAST, "IR"},
    {VLogLevel::OPTIMIZE, VLogLevel::OPTIMIZE_LAST, "optimize"},
    {VLogLevel::PROFILER, VLogLevel::PROFILER_OTHER_0, "profiler"},
  };
  static constexpr VLogTagDesc kVLogTagDescs[] = {
    {VLogLevel::RUNTIME, "VL_RUNTIME", "runtime module base log level"},
    {VLogLevel::RUNTIME_DETAIL, "VL_RUNTIME_DETAIL", "runtime detail log level"},
    {VLogLevel::RUNTIME_MEMORY, "VL_RUNTIME_MEMORY", "runtime memory log level"},
    {VLogLevel::RUNTIME_PIPELINE, "VL_RUNTIME_PIPELINE", "runtime pipeline log level"},
    {VLogLevel::RUNTIME_EXECUTOR, "VL_RUNTIME_EXECUTOR", "runtime executor log level"},
    {VLogLevel::RUNTIME_BUILDER, "VL_RUNTIME_BUILDER", "runtime builder log level"},
    {VLogLevel::RUNTIME_CAPTURE, "VL_RUNTIME_CAPTURE", "runtime capture log level"},
    {VLogLevel::RUNTIME_OTHER, "VL_RUNTIME_OTHER", "runtime other log level"},
    {VLogLevel::OPS, "VL_OPS", "ops module base log level"},
    {VLogLevel::OPS_ACLNN, "VL_OPS_ACLNN", "ops aclnn log level"},
    {VLogLevel::OPS_ATB, "VL_OPS_ATB", "ops atb log level"},
    {VLogLevel::OPS_HCCL, "VL_OPS_HCCL", "ops hccl log level"},
    {VLogLevel::OPS_DVM, "VL_OPS_DVM", "ops dvm log level"},
    {VLogLevel::OPS_CPU, "VL_OPS_CPU", "ops cpu log level"},
    {VLogLevel::OPS_CUSTOM, "VL_OPS_CUSTOM", "ops custom log level"},
    {VLogLevel::OPS_OTHER, "VL_OPS_OTHER", "ops other log level"},
    {VLogLevel::HARDWARE, "VL_HARDWARE", "hardware module base log level"},
    {VLogLevel::HARDWARE_ASCEND, "VL_HARDWARE_ASCEND", "hardware ascend log level"},
    {VLogLevel::HARDWARE_CPU, "VL_HARDWARE_CPU", "hardware cpu log level"},
    {VLogLevel::HARDWARE_MEMORY, "VL_HARDWARE_MEMORY", "hardware memory log level"},
    {VLogLevel::HARDWARE_STREAM, "VL_HARDWARE_STREAM", "hardware stream log level"},
    {VLogLevel::HARDWARE_COLLECTIVE, "VL_HARDWARE_COLLECTIVE", "hardware collective log level"},
    {VLogLevel::HARDWARE_CAPTURE, "VL_HARDWARE_CAPTURE", "hardware capture log level"},
    {VLogLevel::HARDWARE_OTHER, "VL_HARDWARE_OTHER", "hardware other log level"},
    {VLogLevel::COMMON, "VL_COMMON", "common module base log level"},
    {VLogLevel::COMMON_LOADER, "VL_COMMON_LOADER", "common loader log level"},
    {VLogLevel::COMMON_LOGGER, "VL_COMMON_LOGGER", "common logger log level"},
    {VLogLevel::COMMON_UTILS, "VL_COMMON_UTILS", "common utils log level"},
    {VLogLevel::COMMON_OTHER_0, "VL_COMMON_OTHER_0", "common reserved log level 0"},
    {VLogLevel::COMMON_OTHER_1, "VL_COMMON_OTHER_1", "common reserved log level 1"},
    {VLogLevel::COMMON_OTHER_2, "VL_COMMON_OTHER_2", "common reserved log level 2"},
    {VLogLevel::COMMON_OTHER, "VL_COMMON_OTHER", "common other log level"},
    {VLogLevel::CONFIG, "VL_CONFIG", "config module base log level"},
    {VLogLevel::CONFIG_ASCEND, "VL_CONFIG_ASCEND", "config ascend log level"},
    {VLogLevel::CONFIG_ACLGRAPH, "VL_CONFIG_ACLGRAPH", "config aclgraph log level"},
    {VLogLevel::CONFIG_OP_PRECISION, "VL_CONFIG_OP_PRECISION", "config op precision log level"},
    {VLogLevel::CONFIG_OTHER_0, "VL_CONFIG_OTHER_0", "config reserved log level 0"},
    {VLogLevel::CONFIG_OTHER_1, "VL_CONFIG_OTHER_1", "config reserved log level 1"},
    {VLogLevel::CONFIG_OTHER_2, "VL_CONFIG_OTHER_2", "config reserved log level 2"},
    {VLogLevel::CONFIG_OTHER, "VL_CONFIG_OTHER", "config other log level"},
    {VLogLevel::IR, "VL_IR", "IR module base log level"},
    {VLogLevel::IR_GRAPH, "VL_IR_GRAPH", "IR graph log level"},
    {VLogLevel::IR_TENSOR, "VL_IR_TENSOR", "IR tensor log level"},
    {VLogLevel::IR_VALUE, "VL_IR_VALUE", "IR value log level"},
    {VLogLevel::IR_SYMBOLIC, "VL_IR_SYMBOLIC", "IR symbolic log level"},
    {VLogLevel::IR_DTYPE, "VL_IR_DTYPE", "IR dtype log level"},
    {VLogLevel::IR_STORAGE, "VL_IR_STORAGE", "IR storage log level"},
    {VLogLevel::IR_OTHER, "VL_IR_OTHER", "IR other log level"},
    {VLogLevel::OPTIMIZE, "VL_OPTIMIZE", "optimize module base log level"},
    {VLogLevel::OPTIMIZE_PASS, "VL_OPTIMIZE_PASS", "optimize pass log level"},
    {VLogLevel::OPTIMIZE_UD, "VL_OPTIMIZE_UD", "optimize UD log level"},
    {VLogLevel::OPTIMIZE_OTHER_0, "VL_OPTIMIZE_OTHER_0", "optimize reserved log level 0"},
    {VLogLevel::OPTIMIZE_OTHER_1, "VL_OPTIMIZE_OTHER_1", "optimize reserved log level 1"},
    {VLogLevel::OPTIMIZE_OTHER_2, "VL_OPTIMIZE_OTHER_2", "optimize reserved log level 2"},
    {VLogLevel::OPTIMIZE_OTHER_3, "VL_OPTIMIZE_OTHER_3", "optimize reserved log level 3"},
    {VLogLevel::OPTIMIZE_OTHER, "VL_OPTIMIZE_OTHER", "optimize other log level"},
    {VLogLevel::PROFILER, "VL_PROFILER", "profiler module base log level"},
    {VLogLevel::PROFILER_TRACE, "VL_PROFILER_TRACE", "profiler trace log level"},
    {VLogLevel::PROFILER_RUNTIME, "VL_PROFILER_RUNTIME", "profiler runtime log level"},
    {VLogLevel::PROFILER_OPS, "VL_PROFILER_OPS", "profiler ops log level"},
    {VLogLevel::PROFILER_MEMORY, "VL_PROFILER_MEMORY", "profiler memory log level"},
    {VLogLevel::PROFILER_OTHER_0, "VL_PROFILER_OTHER_0", "profiler reserved log level 0"},
    {VLogLevel::FLOW, "VL_FLOW", "flow vlog level"},
    {VLogLevel::DISP_VLOG_TAGS, "VL_DISP_VLOG_TAGS", "log level for printing vlog tags already been used"},
  };

  // This prints the reserved VLOG tag map itself. It is useful when the user sets VLOG_v to 63
  // to discover which ranges are available in this build.
  PrintVLogTag(__FILE__, __LINE__, __FUNCTION__,
               "VLOG usage: export VLOG_v=\"0,2-3,8-15\" to enable individual levels and inclusive ranges.");
  PrintVLogTag(__FILE__, __LINE__, __FUNCTION__, "VLOG module ranges:");
  for (const auto &range : kVLogRangeDescs) {
    std::stringstream ss;
    ss << static_cast<int>(range.begin) << "-" << static_cast<int>(range.end) << ": " << range.module
       << " module vlog levels";
    PrintVLogTag(__FILE__, __LINE__, __FUNCTION__, ss.str());
  }
  PrintVLogTag(__FILE__, __LINE__, __FUNCTION__, "VLOG tags:");
  for (const auto &tag : kVLogTagDescs) {
    std::stringstream ss;
    ss << static_cast<int>(tag.level) << ": " << tag.name << " - " << tag.description;
    PrintVLogTag(__FILE__, __LINE__, __FUNCTION__, ss.str());
  }
}

void InitLogConfig() {
  auto &config = GetConfig();

  int global_log_level = kDefaultLogLevel;
  const auto glog_v = GetEnv("GLOG_v");
  if (glog_v.empty()) {
#ifdef USE_GLOG
    FLAGS_v = kDefaultLogLevel;
#endif
  } else if (ParseLogLevel(glog_v, &global_log_level)) {
#ifdef USE_GLOG
    FLAGS_v = global_log_level;
#endif
  } else {
    OutputWarningBeforeInit("Value of environment var GLOG_v is invalid: " + glog_v);
  }
  config.glog_level = global_log_level;

  const auto vlog_v = GetEnv("VLOG_v");
  if (!vlog_v.empty()) {
    uint64_t vlog_mask = 0;
    if (ParseVLogMask(vlog_v, &vlog_mask)) {
      g_mrt_vlog_mask = vlog_mask;
    } else {
      g_mrt_vlog_mask = 0;
      OutputWarningBeforeInit("Value of environment var VLOG_v is invalid: " + vlog_v);
    }
  }

#ifdef USE_GLOG
  FLAGS_log_prefix = false;
  FLAGS_logbufsecs = 0;
  if (GetEnv("GLOG_logfile_mode").empty()) {
    FLAGS_logfile_mode = 0640;
  }
  FLAGS_max_log_size = 50;
  const auto max_log_size = GetEnv("GLOG_max_log_size");
  if (!max_log_size.empty()) {
    try {
      auto parsed_size = std::stoi(max_log_size);
      if (parsed_size > 0 && parsed_size <= INT32_MAX) {
        FLAGS_max_log_size = static_cast<decltype(FLAGS_max_log_size)>(parsed_size);
      } else {
        OutputWarningBeforeInit("Invalid GLOG_max_log_size value: " + max_log_size + ". Using default 50 MB.");
      }
    } catch (const std::exception &e) {
      OutputWarningBeforeInit("Invalid GLOG_max_log_size value: " + max_log_size + ". Using default 50 MB.");
    }
  }

  FLAGS_logtostderr = true;
  const auto logtostderr = GetEnv("GLOG_logtostderr");
  if (logtostderr == "0") {
    const auto log_dir = GetEnv("GLOG_log_dir");
    if (log_dir.empty()) {
      OutputWarningBeforeInit("`GLOG_log_dir` is empty, fallback to stderr.");
    } else {
      FLAGS_logtostderr = false;
      FLAGS_log_dir = GetRealLogPath(log_dir);
      auto path = FLAGS_log_dir;
      if (!MakePath(path.data())) {
        OutputWarningBeforeInit("Failed to create log path " + FLAGS_log_dir + ", fallback to stderr.");
        FLAGS_logtostderr = true;
      }
    }
  }
  FLAGS_stderrthreshold = GetThresholdLevel(GetEnv("GLOG_stderrthreshold"));

  if (!mrt_private::IsGoogleLoggingInitialized()) {
    mrt_private::InitGoogleLogging("inferrt");
  }
#endif
  DispVLogTags();
}

void EnsureLogInitialized() { std::call_once(GetInitFlag(), InitLogConfig); }

struct LogInitializer {
  LogInitializer() { EnsureLogInitialized(); }
};

LogInitializer g_log_initializer;

std::string NormalizeFileName(const char *file) {
  if (file == nullptr || file[0] == '\0') {
    return "";
  }

  std::string file_name(file);
  auto pos = file_name.rfind("/inferrt/");
  if (pos != std::string::npos) {
    return file_name.substr(pos + 1);
  }

  pos = file_name.find("inferrt/");
  if (pos != std::string::npos) {
    return file_name.substr(pos);
  }
  return file_name;
}

std::string BuildExceptionMessage(const std::string &message, const std::string &file_name, int line) {
  std::stringstream ss;
  ss << message;
  if (!file_name.empty()) {
    ss << "\n" << kSplitLine << "- C++ Call Stack: (For framework developers) \n" << kSplitLine;
    ss << file_name << "(" << line << ").\n\n";
  }
  return ss.str();
}

void OutputLogMessage(const char *file, int line, const char *func, GLogLevel level, VLogLevel vlog_level, bool is_vlog,
                      const std::string &message) {
  const std::string file_name = NormalizeFileName(file);
  const auto vlog_level_value = static_cast<int>(vlog_level);
#ifdef USE_GLOG
  mrt_private::LogMessage("", 0, GetGlogLevel(level)).stream()
    << "[" << (is_vlog ? "V" + std::to_string(vlog_level_value) : GetLogLevelName(level)) << "] " << GetTimeString()
    << " InferRT [pid:" << getpid() << ", thread id:" << std::hex << std::this_thread::get_id() << std::dec << " "
    << file_name << ":" << line << " " << func << "] " << message << std::endl;
#else
  std::cerr << "[" << (is_vlog ? "V" + std::to_string(vlog_level_value) : GetLogLevelName(level)) << "] "
            << GetTimeString() << " InferRT [pid:" << getpid() << ", thread id:" << std::hex
            << std::this_thread::get_id() << std::dec << " " << file_name << ":" << line << " " << func << "] "
            << message << std::endl;
#endif
}
}  // namespace

LogWriter::LogWriter(const char *file, int line, const char *func, GLogLevel level)
    : file_(file), line_(line), func_(func), level_(level), vlog_level_(VLogLevel::DISP_VLOG_TAGS), is_vlog_(false) {}

LogWriter::LogWriter(const char *file, int line, const char *func, VLogLevel vlog_level)
    : file_(file), line_(line), func_(func), level_(GLogLevel::INFO), vlog_level_(vlog_level), is_vlog_(true) {}

void LogWriter::operator<(const LogStream &stream) const {
  const bool should_throw = IsExceptionLevel(level_);
  std::string message;
  try {
    message = stream.Stream().str();
    if (should_throw) {
      message = BuildExceptionMessage(message, NormalizeFileName(file_), line_);
    }
    OutputLogMessage(file_, line_, func_, level_, vlog_level_, is_vlog_, message);
  } catch (...) {
    if (!should_throw) {
      return;
    }
    message = BuildExceptionMessage("Exception occurred while formatting or writing log message.",
                                    NormalizeFileName(file_), line_);
  }
  if (should_throw) {
    throw std::runtime_error(message);
  }
}

bool IsGlogOn(GLogLevel level) { return static_cast<int>(level) >= GetConfig().glog_level; }
}  // namespace common
}  // namespace mrt
