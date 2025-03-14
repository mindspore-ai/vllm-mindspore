#ifndef VLLM_MINDSPORE_OPS_ASCENDC_ADV_STEP_FLASH_H
#define VLLM_MINDSPORE_OPS_ASCENDC_ADV_STEP_FLASH_H

extern void AdvStepFlashKernelEntry(uint32_t blockDims, void *l2ctrl, void *aclStream, uint8_t *sampledTokenIds,
                                    uint8_t *blockTables, uint8_t *seqLensInput, uint8_t *inputTokens,
                                    uint8_t *inputPositions, uint8_t *seqLensOut, uint8_t *slotMapping,
                                    int32_t num_seqs, int32_t block_size, int32_t block_tables_stride);

#endif  // VLLM_MINDSPORE_OPS_ASCENDC_ADV_STEP_FLASH_H
