/**
 * Copyright 2025 Huawei Technologies Co., Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef VLLM_MINDSPORE_OPS_ASCENDC_ADV_STEP_FLASH_H
#define VLLM_MINDSPORE_OPS_ASCENDC_ADV_STEP_FLASH_H

extern void AdvStepFlashKernelEntry(
    uint32_t blockDims, void *l2ctrl, void *aclStream, uint8_t *sampledTokenIds,
    uint8_t *blockTables, uint8_t *seqLensInput, uint8_t *inputTokens,
    uint8_t *inputPositions, uint8_t *seqLensOut, uint8_t *slotMapping,
    int32_t num_seqs, int32_t block_size, int32_t block_tables_stride);

#endif // VLLM_MINDSPORE_OPS_ASCENDC_ADV_STEP_FLASH_H
