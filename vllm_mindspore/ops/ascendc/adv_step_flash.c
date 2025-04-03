#include "kernel_operator.h"

using namespace AscendC;

template <typename Tp, Tp v>
struct integral_constant {
  static constexpr Tp value = v;
};
using true_type = integral_constant<bool, true>;
using false_type = integral_constant<bool, false>;
template <typename, typename>
struct is_same : public false_type {};
template <typename Tp>
struct is_same<Tp, Tp> : public true_type {};

template <typename T, typename U, typename R>
__aicore__ inline void DataCopyCustom(const U &dstTensor, const R &srcTensor, const uint32_t count) {
  DataCopyParams copyParams;
  copyParams.blockLen = count * sizeof(T);
  copyParams.blockCount = 1;
  if constexpr (is_same<U, AscendC::LocalTensor<T>>::value) {
    DataCopyPadParams padParams;
    DataCopyPad(dstTensor, srcTensor, copyParams, padParams);
  } else {
    DataCopyPad(dstTensor, srcTensor, copyParams);
  }
}

class KernelAdvStepFlash {
 public:
  __aicore__ inline KernelAdvStepFlash(TPipe *pipe) { Ppipe = pipe; }

  __aicore__ inline void Init(GM_ADDR sampledTokenIds, GM_ADDR blockTables, GM_ADDR seqLensInput, GM_ADDR inputTokens,
                              GM_ADDR inputPositions, GM_ADDR seqLensOut, GM_ADDR slotMapping, int32_t num_seqs,
                              int32_t block_size, int32_t block_tables_stride) {
    ASSERT(GetBlockNum() != 0 && "Block dim can not be zero!");
    this->blockSize = block_size;
    this->blockTablesStride = block_tables_stride;
    this->tensorLength = num_seqs;

    this->blockSizeFp = static_cast<float>(this->blockSize);

    // get start index for current core, core parallel
    sampledTokenIdsGm.SetGlobalBuffer((__gm__ int32_t *)sampledTokenIds, tensorLength);
    seqLensInputGm.SetGlobalBuffer((__gm__ int32_t *)seqLensInput, tensorLength);
    blockTablesGm.SetGlobalBuffer((__gm__ int32_t *)blockTables);  // inf size

    inputTokensGm.SetGlobalBuffer((__gm__ int32_t *)inputTokens, tensorLength);
    inputPositionsGm.SetGlobalBuffer((__gm__ int32_t *)inputPositions, tensorLength);
    seqLensOutGm.SetGlobalBuffer((__gm__ int32_t *)seqLensOut, tensorLength);
    slotMappingGm.SetGlobalBuffer((__gm__ int32_t *)slotMapping, tensorLength);

    // pipe alloc memory to queue, the unit is Bytes
    Ppipe->InitBuffer(sampledIdsQue, 1, tensorLength * sizeof(int32_t));
    Ppipe->InitBuffer(seqLenInQue, 1, tensorLength * sizeof(int32_t));

    Ppipe->InitBuffer(inputTokensQue, 1, tensorLength * sizeof(int32_t));
    Ppipe->InitBuffer(seqLensOutQue, 1, tensorLength * sizeof(int32_t));
    Ppipe->InitBuffer(inputPositionsQue, 1, tensorLength * sizeof(int32_t));

    Ppipe->InitBuffer(tableOffsetBuf, tensorLength * sizeof(int32_t));

    Ppipe->InitBuffer(tmpDivBuf01, tensorLength * sizeof(int32_t));
    Ppipe->InitBuffer(tmpDivBuf02, tensorLength * sizeof(int32_t));

    Ppipe->InitBuffer(outTableBuf, tensorLength * sizeof(int32_t));
    Ppipe->InitBuffer(blockTableBuf, 32);
  }

  __aicore__ inline void Process() {
    CopyIn();
    Compute();
    CopyOut();
  }

 private:
  __aicore__ inline void CopyIn() {
    LocalTensor<int32_t> sampledIdsLocal = sampledIdsQue.AllocTensor<int32_t>();
    LocalTensor<int32_t> seqLenInLocal = seqLenInQue.AllocTensor<int32_t>();

    DataCopyCustom<int32_t>(sampledIdsLocal, sampledTokenIdsGm, tensorLength);
    DataCopyCustom<int32_t>(seqLenInLocal, seqLensInputGm, tensorLength);

    sampledIdsQue.EnQue(sampledIdsLocal);
    seqLenInQue.EnQue(seqLenInLocal);
  }

  __aicore__ inline void Compute() {
    LocalTensor<int32_t> tableOffset = tableOffsetBuf.Get<int32_t>();

    LocalTensor<int32_t> sampledIdsLocal = sampledIdsQue.DeQue<int32_t>();
    LocalTensor<int32_t> seqLenInLocal = seqLenInQue.DeQue<int32_t>();

    LocalTensor<int32_t> inputTokensLocal = inputTokensQue.AllocTensor<int32_t>();
    LocalTensor<int32_t> seqLensOutLocal = seqLensOutQue.AllocTensor<int32_t>();
    LocalTensor<int32_t> inputPositionsLocal = inputPositionsQue.AllocTensor<int32_t>();

    Adds(inputTokensLocal, sampledIdsLocal, (int32_t)0, tensorLength);   // inputTokensLocal <-- sampledIdsLocal
    Adds(inputPositionsLocal, seqLenInLocal, (int32_t)0, tensorLength);  // inputPositionsLocal <-- seqLenInLocal
    Adds(seqLensOutLocal, seqLenInLocal, (int32_t)1, tensorLength);      // seqLensOutLocal <-- seqLenInLocal + 1
    PipeBarrier<PIPE_V>();

    // TODO add Function
    ComputeTableOffset(tableOffset, inputPositionsLocal);
    // GetTableValueByOffset(tableOffset, inputPositionsLocal);

    sampledIdsQue.FreeTensor(sampledIdsLocal);
    seqLenInQue.FreeTensor(seqLenInLocal);

    inputTokensQue.EnQue(inputTokensLocal);
    seqLensOutQue.EnQue(seqLensOutLocal);
    inputPositionsQue.EnQue(inputPositionsLocal);
  }

  __aicore__ inline void CopyOut() {
    LocalTensor<int32_t> inputTokensLocal = inputTokensQue.DeQue<int32_t>();
    LocalTensor<int32_t> seqLensOutLocal = seqLensOutQue.DeQue<int32_t>();
    LocalTensor<int32_t> inputPositionsLocal = inputPositionsQue.DeQue<int32_t>();

    DataCopyCustom<int32_t>(inputTokensGm, inputTokensLocal, tensorLength);
    DataCopyCustom<int32_t>(inputPositionsGm, inputPositionsLocal, tensorLength);
    DataCopyCustom<int32_t>(seqLensOutGm, seqLensOutLocal, tensorLength);

    inputTokensQue.FreeTensor(inputTokensLocal);
    seqLensOutQue.FreeTensor(seqLensOutLocal);
    inputPositionsQue.FreeTensor(inputPositionsLocal);
  }

  __aicore__ inline void ComputeTableOffset(LocalTensor<int32_t> tableOffset,
                                            LocalTensor<int32_t> inputPositionsLocal) {
    LocalTensor<float> tmpBuf01 = tmpDivBuf01.Get<float>();
    LocalTensor<float> tmpBuf02 = tmpDivBuf02.Get<float>();

    LocalTensor<int32_t> tmpBuf01Int = tmpBuf01.ReinterpretCast<int32_t>();
    LocalTensor<int32_t> tmpBuf02Int = tmpBuf02.ReinterpretCast<int32_t>();

    LocalTensor<int32_t> outTableValue = outTableBuf.Get<int32_t>();
    LocalTensor<int32_t> blockTableLocal = blockTableBuf.Get<int32_t>();

    // floor div
    Cast(tmpBuf01, inputPositionsLocal, RoundMode::CAST_RINT, tensorLength);
    Duplicate(tmpBuf02, blockSizeFp, tensorLength);
    PipeBarrier<PIPE_V>();
    Div(tmpBuf01, tmpBuf01, tmpBuf02, tensorLength);  // <-- inputPositionsLocal / blockSize
    PipeBarrier<PIPE_V>();
    Cast(tmpBuf02Int, tmpBuf01, RoundMode::CAST_TRUNC, tensorLength);

    CreateVecIndex(tableOffset, (int32_t)0, tensorLength);  // tableOffset <--- 0, 1, 2, 3, .... tensorLength -1
    PipeBarrier<PIPE_V>();

    Muls(tableOffset, tableOffset, this->blockTablesStride,
         tensorLength);  // tableOffset <--- curt_offset * block_stride
    PipeBarrier<PIPE_V>();
    Add(tableOffset, tableOffset, tmpBuf02Int,
        tensorLength);  // tableOffset <--- curt_offset * block_stride + inputPositionsLocal / blockSize

    PIPE_V_S();

    for (int32_t idx = 0; idx < tensorLength; idx++) {
      int32_t blockTableIdx = tableOffset.GetValue(idx);

      PIPE_S_MTE2();

      DataCopyCustom<int32_t>(blockTableLocal, blockTablesGm[blockTableIdx], 1);

      PIPE_MTE2_S();

      int32_t blockTableValue = blockTableLocal.GetValue(0);
      int32_t block_offset = inputPositionsLocal.GetValue(idx) % this->blockSize;
      blockTableValue = blockTableValue * this->blockSize + block_offset;
      outTableValue.SetValue(idx, blockTableValue);
    }
    PIPE_S_MTE3();
    DataCopyCustom<int32_t>(slotMappingGm, outTableValue, tensorLength);
  }

  __aicore__ inline void PIPE_S_MTE3() {
    event_t event_S_MTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(event_S_MTE3);
    WaitFlag<HardEvent::S_MTE3>(event_S_MTE3);
  }

  __aicore__ inline void PIPE_S_MTE2() {
    event_t event_S_MTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE2));
    SetFlag<HardEvent::S_MTE2>(event_S_MTE2);
    WaitFlag<HardEvent::S_MTE2>(event_S_MTE2);
  }

  __aicore__ inline void PIPE_MTE2_S() {
    event_t event_MTE2_S = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(event_MTE2_S);
    WaitFlag<HardEvent::MTE2_S>(event_MTE2_S);
  }

  __aicore__ inline void PIPE_V_S() {
    event_t event_V_S = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(event_V_S);
    WaitFlag<HardEvent::V_S>(event_V_S);
  }

 private:
  TPipe *Ppipe = nullptr;
  // create queues for input, in this case depth is equal to buffer num
  TQue<QuePosition::VECIN, 1> sampledIdsQue, seqLenInQue;
  // create queues for output, in this case depth is equal to buffer num
  TQue<QuePosition::VECOUT, 1> inputTokensQue, seqLensOutQue, inputPositionsQue;

  TBuf<TPosition::VECCALC> tableOffsetBuf;
  TBuf<TPosition::VECCALC> tmpDivBuf01;
  TBuf<TPosition::VECCALC> tmpDivBuf02;
  TBuf<TPosition::VECCALC> outTableBuf;
  TBuf<TPosition::VECCALC> blockTableBuf;

  // inputs
  GlobalTensor<int32_t> sampledTokenIdsGm;
  GlobalTensor<int32_t> seqLensInputGm;
  GlobalTensor<int32_t> blockTablesGm;
  // outs
  GlobalTensor<int32_t> inputTokensGm;
  GlobalTensor<int32_t> inputPositionsGm;
  GlobalTensor<int32_t> seqLensOutGm;
  GlobalTensor<int32_t> slotMappingGm;

  int32_t blockSize;
  int32_t blockTablesStride;
  int64_t tensorLength;  // number of calculations rows on each core

  float blockSizeFp;
};

extern "C" __global__ __aicore__ void adv_step_flash_impl(GM_ADDR sampledTokenIds, GM_ADDR blockTables,
                                                          GM_ADDR seqLensInput, GM_ADDR inputTokens,
                                                          GM_ADDR inputPositions, GM_ADDR seqLensOut,
                                                          GM_ADDR slotMapping, int32_t num_seqs, int32_t block_size,
                                                          int32_t block_tables_stride) {
  TPipe pipe;

  KernelAdvStepFlash op(&pipe);
  op.Init(sampledTokenIds, blockTables, seqLensInput, inputTokens, inputPositions, seqLensOut, slotMapping, num_seqs,
          block_size, block_tables_stride);
  op.Process();
}

#ifndef __CCE_KT_TEST__
void AdvStepFlashKernelEntry(uint32_t blockDims, void *l2ctrl, void *aclStream, uint8_t *sampledTokenIds,
                             uint8_t *blockTables, uint8_t *seqLensInput, uint8_t *inputTokens, uint8_t *inputPositions,
                             uint8_t *seqLensOut, uint8_t *slotMapping, int32_t num_seqs, int32_t block_size,
                             int32_t block_tables_stride) {
  adv_step_flash_impl<<<blockDims, l2ctrl, aclStream>>>(sampledTokenIds, blockTables, seqLensInput, inputTokens,
                                                        inputPositions, seqLensOut, slotMapping, num_seqs, block_size,
                                                        block_tables_stride);
}
#endif
