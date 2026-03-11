/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include <nanobind/stl/string.h>

#include "ops/custom_op_register.h"
#include "profiler/profiler.h"

namespace nb = nanobind;

// Interface with python
NB_MODULE(_ms_inferrt_api, mod) {
  mod.def(
    "is_custom_op_registered",
    [](const std::string &op_name) { return mrt::ops::CustomOpRegistry::GetInstance().IsCustomOpRegistered(op_name); },
    nb::arg("op_name"), "Check if a custom operator is registered.");

  // Profiler functions
  mod.def(
    "mrt_profiler_start_step", []() { mrt::profiler::ProfilerAnalyzer::GetInstance().StartStep(); },
    "Start a profiling step.");
  mod.def(
    "mrt_profiler_end_step", []() { mrt::profiler::ProfilerAnalyzer::GetInstance().EndStep(); },
    "End a profiling step.");
  mod.def(
    "mrt_profiler_clear", []() { mrt::profiler::ProfilerAnalyzer::GetInstance().Clear(); },
    "Clear all profiling data.");
  mod.def(
    "mrt_profiler_enable", []() { mrt::profiler::ProfilerAnalyzer::GetInstance().Enable(); }, "Enable the profiler.");
  mod.def(
    "mrt_profiler_disable", []() { mrt::profiler::ProfilerAnalyzer::GetInstance().Disable(); },
    "Disable the profiler.");
  mod.def(
    "mrt_profiler_is_enabled", []() { return mrt::profiler::ProfilerAnalyzer::GetInstance().IsProfilerEnabled(); },
    "Check if the profiler is enabled.");
}
