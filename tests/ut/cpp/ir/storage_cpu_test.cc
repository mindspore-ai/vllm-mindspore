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

#include "gtest/gtest.h"
#include "ir/tensor/storage.h"

namespace mrt {
using Storage = ir::Storage;

class StorageTest : public testing::Test {
 protected:
  void SetUp() override {
    dataSize = 1024;
    data = malloc(dataSize);
  }

  void TearDown() override {
    if (data != nullptr) {
      free(data);
      data = nullptr;
    }
  }

  void *data;
  size_t dataSize;
  hardware::Device device{hardware::DeviceType::CPU};
};

/// Feature: Storage
/// Description: Initialize Storage with size and device
/// Expectation: Storage created with correct properties
TEST_F(StorageTest, TestConstructor) {
  Storage storage(dataSize, device);
  EXPECT_EQ(storage.SizeBytes(), dataSize);
  EXPECT_EQ(storage.GetDevice().type, device.type);
  EXPECT_TRUE(storage.CheckCanOwnData());
  EXPECT_EQ(storage.Data(), nullptr);
}

/// Feature: Storage
/// Description: Initialize Storage with external data pointer
/// Expectation: Storage references external data without ownership
TEST_F(StorageTest, TestConstructorWithNonOwnedData) {
  Storage storage(data, dataSize, device);
  EXPECT_EQ(storage.SizeBytes(), dataSize);
  EXPECT_EQ(storage.GetDevice().type, device.type);
  EXPECT_FALSE(storage.CheckCanOwnData());
  EXPECT_EQ(storage.Data(), data);
}

/// Feature: Storage
/// Description: Resize Storage capacity
/// Expectation: Size updated without affecting data ownership
TEST_F(StorageTest, TestResize) {
  Storage storage(dataSize, device);
  storage.Resize(2048);
  EXPECT_EQ(storage.SizeBytes(), 2048);
  EXPECT_EQ(storage.Data(), nullptr);
}

/// Feature: Storage
/// Description: Allocate and deallocate device memory
/// Expectation: Memory properly allocated and freed
TEST_F(StorageTest, TestAllocate) {
  Storage storage(dataSize, device);
  EXPECT_EQ(storage.Data(), nullptr);
  storage.AllocateMemory();
  EXPECT_NE(storage.Data(), nullptr);

  storage.FreeMemory();
  EXPECT_EQ(storage.Data(), nullptr);
}

/// Feature: Storage
/// Description: Attempt to allocate memory twice
/// Expectation: Second allocation throws runtime_error
TEST_F(StorageTest, TestDoubleAllocate) {
  Storage storage(dataSize, device);
  storage.AllocateMemory();

  void *dataPtr = storage.Data();
  EXPECT_THROW(storage.AllocateMemory(), std::runtime_error);
  EXPECT_EQ(storage.Data(), dataPtr);
}

/// Feature: Storage
/// Description: Attempt to free non-owned data
/// Expectation: Free operation throws runtime_error
TEST_F(StorageTest, TestFreeNonOwnedData) {
  Storage storage(data, dataSize, device);
  EXPECT_THROW(storage.FreeMemory(), std::runtime_error);
  EXPECT_EQ(storage.Data(), data);
}

/// Feature: Storage
/// Description: Release data ownership from Storage
/// Expectation: Ownership released and data pointer set to null
TEST_F(StorageTest, TestRelease) {
  Storage storage(dataSize, device);
  storage.AllocateMemory();

  void *dataPtr = storage.Data();
  void *releasePtr = storage.Release();
  EXPECT_EQ(dataPtr, releasePtr);
  EXPECT_EQ(storage.Data(), nullptr);

  free(releasePtr);
}

/// Feature: Storage
/// Description: Destroy Storage referencing external data
/// Expectation: External data remains valid after Storage destruction
TEST_F(StorageTest, TestDestructorWithNonOwnedData) {
  void *dataPtr = malloc(512);
  { Storage storage(dataPtr, 512, device); }
  EXPECT_NO_THROW(free(dataPtr));
}

}  // namespace mrt
