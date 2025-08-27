/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef __COMMON_VISIBLE_H__
#define __COMMON_VISIBLE_H__

#if (defined(_WIN32) || defined(__WIN32__) || defined(WIN32) || defined(__CYGWIN__))
#define DA_API __declspec(dllexport)
#else
#define DA_API __attribute__((visibility("default")))
#endif

#if (defined(_WIN32) || defined(__WIN32__) || defined(WIN32) || defined(__CYGWIN__))
#ifdef HARDWARE_DLL
#define MRT_EXPORT __declspec(dllexport)
#else
#define MRT_EXPORT __declspec(dllimport)
#endif
#define MRT_LOCAL
#else
#define MRT_EXPORT __attribute__((visibility("default")))
#define MRT_LOCAL __attribute__((visibility("hidden")))
#endif
#endif  // __COMMON_VISIBLE_H__
