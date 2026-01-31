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

#include <chrono>
#include <thread>
#include <iostream>
#include "gtest/gtest.h"
#include "profiler/profiler.h"

using mrt::profiler::ProfilerAnalyzer;
using mrt::profiler::ProfilerEvent;
using mrt::profiler::ProfilerModule;
using mrt::profiler::ProfilerRecorder;
using mrt::profiler::ProfilerStage;
using mrt::profiler::ProfilerStageRecorder;

class ProfilerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Ensure profiler is disabled before test starts
    ProfilerAnalyzer::GetInstance().Disable();
    ProfilerAnalyzer::GetInstance().Clear();
  }

  void TearDown() override {
    // Clean up resources after test ends
    ProfilerAnalyzer::GetInstance().Disable();
    ProfilerAnalyzer::GetInstance().Clear();
  }
};

// Test basic functionality: enable/disable
TEST_F(ProfilerTest, BasicFunctionality) {
  // Initial state should be disabled
  EXPECT_FALSE(ProfilerAnalyzer::GetInstance().IsProfilerEnabled());

  // Test enable functionality
  ProfilerAnalyzer::GetInstance().Enable();
  EXPECT_TRUE(ProfilerAnalyzer::GetInstance().IsProfilerEnabled());

  // Test disable functionality
  ProfilerAnalyzer::GetInstance().Disable();
  EXPECT_FALSE(ProfilerAnalyzer::GetInstance().IsProfilerEnabled());
}

// Test timestamp acquisition
TEST_F(ProfilerTest, TimeStamp) {
  uint64_t ts1 = ProfilerAnalyzer::GetInstance().GetTimeStamp();
  // Wait for a short time
  std::this_thread::sleep_for(std::chrono::milliseconds(1));
  uint64_t ts2 = ProfilerAnalyzer::GetInstance().GetTimeStamp();

  // Ensure timestamp is increasing
  EXPECT_GT(ts2, ts1);
}

// Test singleton pattern
TEST_F(ProfilerTest, Singleton) {
  ProfilerAnalyzer &instance1 = ProfilerAnalyzer::GetInstance();
  ProfilerAnalyzer &instance2 = ProfilerAnalyzer::GetInstance();

  // Ensure it's the same instance
  EXPECT_EQ(&instance1, &instance2);
}

// Test ProfilerRecorder RAII functionality
TEST_F(ProfilerTest, ProfilerRecorder) {
  // Enable profiler
  ProfilerAnalyzer::GetInstance().Enable();
  EXPECT_TRUE(ProfilerAnalyzer::GetInstance().IsProfilerEnabled());

  // Start a step
  ProfilerAnalyzer::GetInstance().StartStep();

  // Use RAII recorder to record event
  {
    ProfilerRecorder recorder(ProfilerModule::kRuntime, ProfilerEvent::kDefault, "TestOp", false);
    // Simulate workload
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  // End the step
  ProfilerAnalyzer::GetInstance().EndStep();

  // Disable profiler
  ProfilerAnalyzer::GetInstance().Disable();
}

// Test ProfilerStageRecorder RAII functionality
TEST_F(ProfilerTest, ProfilerStageRecorder) {
  ProfilerAnalyzer::GetInstance().Enable();
  EXPECT_TRUE(ProfilerAnalyzer::GetInstance().IsProfilerEnabled());

  ProfilerAnalyzer::GetInstance().StartStep();

  // Use RAII recorder to record stage
  {
    ProfilerStageRecorder recorder(ProfilerStage::kInfer);
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  // End step
  ProfilerAnalyzer::GetInstance().EndStep();

  // 禁用profiler
  ProfilerAnalyzer::GetInstance().Disable();
}

// Test multiple steps recording
TEST_F(ProfilerTest, MultipleSteps) {
  ProfilerAnalyzer::GetInstance().Enable();
  EXPECT_TRUE(ProfilerAnalyzer::GetInstance().IsProfilerEnabled());

  // Record multiple steps
  for (int i = 0; i < 3; ++i) {
    ProfilerAnalyzer::GetInstance().StartStep();

    // Record some events
    {
      ProfilerStageRecorder recorder(ProfilerStage::kRunGraph);
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    {
      ProfilerRecorder recorder(ProfilerModule::kKernel, ProfilerEvent::kKernelExecute, "TestKernel", false);
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    ProfilerAnalyzer::GetInstance().EndStep();
  }

  ProfilerAnalyzer::GetInstance().Disable();
}

// Test behavior when disabled
TEST_F(ProfilerTest, DisabledBehavior) {
  // Ensure profiler is disabled
  ProfilerAnalyzer::GetInstance().Disable();
  EXPECT_FALSE(ProfilerAnalyzer::GetInstance().IsProfilerEnabled());

  // Start step (when disabled)
  ProfilerAnalyzer::GetInstance().StartStep();

  // Use recorder (should not record any data when disabled)
  {
    ProfilerRecorder recorder(ProfilerModule::kRuntime, ProfilerEvent::kDefault, "TestOp", false);
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  // End step (when disabled)
  ProfilerAnalyzer::GetInstance().EndStep();

  // Clear data
  ProfilerAnalyzer::GetInstance().Clear();
}

// Test data clearing functionality
TEST_F(ProfilerTest, ClearData) {
  ProfilerAnalyzer::GetInstance().Enable();
  EXPECT_TRUE(ProfilerAnalyzer::GetInstance().IsProfilerEnabled());

  // Record some data
  ProfilerAnalyzer::GetInstance().StartStep();
  {
    ProfilerRecorder recorder(ProfilerModule::kRuntime, ProfilerEvent::kDefault, "TestOp", false);
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  ProfilerAnalyzer::GetInstance().EndStep();

  // Clear data
  ProfilerAnalyzer::GetInstance().Clear();

  ProfilerAnalyzer::GetInstance().Disable();
}

// Test environment variable initialization
TEST_F(ProfilerTest, EnvironmentVariable) {
  // Set environment variables to enable profiler
  setenv("MRT_ENABLE_RUNTIME_PROFILER", "1", 1);
  setenv("MRT_ENABLE_PROFILER_TOP_NUM", "20", 1);

  // Reset initialization state
  ProfilerAnalyzer::GetInstance().Disable();
  ProfilerAnalyzer::GetInstance().Reset();

  // Calling StartStep triggers initialization
  ProfilerAnalyzer::GetInstance().StartStep();

  // Check if enabled via environment variable
  EXPECT_TRUE(ProfilerAnalyzer::GetInstance().IsProfilerEnabled());

  // Clean up environment variables
  unsetenv("MRT_ENABLE_RUNTIME_PROFILER");
  unsetenv("MRT_ENABLE_PROFILER_TOP_NUM");

  ProfilerAnalyzer::GetInstance().EndStep();
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
