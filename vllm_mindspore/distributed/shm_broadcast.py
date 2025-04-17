#!/usr/bin/env python3
# encoding: utf-8
# Copyright 2025 Huawei Technologies Co., Ltd
# Copyright 2024 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import numpy as np
from typing import Optional
from multiprocessing import shared_memory
from unittest.mock import patch
from vllm.logger import init_logger

logger = init_logger(__name__)

def initialize_ShmRingBuffer(self,
                            n_reader: int,
                            max_chunk_bytes: int,
                            max_chunks: int,
                            name: Optional[str] = None):
    logger.info("Entering mindspore shm_broadcast")
    self.n_reader = n_reader
    self.metadata_size = 1 + n_reader
    self.max_chunk_bytes = max_chunk_bytes
    self.max_chunks = max_chunks
    self.total_bytes_of_buffer = (self.max_chunk_bytes +
                                    self.metadata_size) * self.max_chunks
    self.data_offset = 0
    self.metadata_offset = self.max_chunk_bytes * self.max_chunks

    if name is None:
        # we are creating a buffer
        self.is_creator = True
        self.shared_memory = shared_memory.SharedMemory(
            create=True, size=self.total_bytes_of_buffer)
        # initialize the metadata section to 0
        with memoryview(self.shared_memory.buf[self.metadata_offset:]
                        ) as metadata_buffer:
            np.frombuffer(metadata_buffer, dtype=np.uint8).fill(0)
    else:
        # we are opening an existing buffer
        self.is_creator = False
        # fix to https://stackoverflow.com/q/62748654/9191338
        # Python incorrectly tracks shared memory even if it is not
        # created by the process. The following patch is a workaround.
        with patch("multiprocessing.resource_tracker.register",
                    lambda *args, **kwargs: None):
            try:
                self.shared_memory = shared_memory.SharedMemory(name=name)
                # See https://docs.python.org/3/library/multiprocessing.shared_memory.html # noqa
                # Some platforms allocate memory based on page size,
                # so the shared memory block size may be larger or equal
                # to the requested size. The size parameter is ignored
                # when attaching to an existing block.
                assert (self.shared_memory.size
                        >= self.total_bytes_of_buffer)
            except FileNotFoundError:
                # we might deserialize the object in a different node
                # in this case, this object is not used,
                # and we should suppress the error
                pass
