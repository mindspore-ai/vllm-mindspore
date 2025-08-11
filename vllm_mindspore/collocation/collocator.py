# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 Huawei Technologies Co., Ltd.
# Copyright 2024 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import mmap
import os
import time
from contextlib import suppress
from typing import Optional

import posix_ipc
from vllm.logger import init_logger

logger = init_logger("vllm.collocation.collocator")


def _init_semaphore(name: str, initial_value: int):
    sem = posix_ipc.Semaphore(name=name,
                              flags=posix_ipc.O_CREAT,
                              initial_value=initial_value)
    while (sem.value > initial_value):
        sem.acquire()
    while (sem.value < initial_value):
        sem.release()
    return sem


def _init_shared_memory(name: str, size: int, initial_value: int):
    sm = posix_ipc.SharedMemory(name=name, flags=posix_ipc.O_CREAT, size=size)
    sm_file = mmap.mmap(sm.fd, sm.size)
    sm_file.seek(0)
    sm_file.write_byte(initial_value)
    sm_file.seek(0)
    sm.close_fd()
    return (sm, sm_file)


class CollocatorDevices:
    '''Class to collocate execution through IPC
    When all devices synchronizes the launch of prefill/decode and logits
    '''

    def __init__(self):
        logger.info("CollocatorDevices: __init__")
        if os.environ.get("MS_ENABLE_LCCL", False) not in ['on']:
            logger.warning("May need to export MS_ENABLE_LCCL=on")
        ms_dev_runtime_conf = os.environ.get("MS_DEV_RUNTIME_CONF", False)
        if (ms_dev_runtime_conf is False
                or "comm_init_lccl_only:true" not in ms_dev_runtime_conf):
            logger.warning("May need to export "
                           "MS_DEV_RUNTIME_CONF=\"comm_init_lccl_only:true\"")
        disable_custom_kernel = os.environ.get(
            "MS_INTERNAL_DISABLE_CUSTOM_KERNEL_LIST", False)
        if (disable_custom_kernel is False
                or "MatMulAllReduce" not in disable_custom_kernel):
            logger.warning(
                "May need to export "
                "MS_INTERNAL_DISABLE_CUSTOM_KERNEL_LIST=MatMulAllReduce")

        self.nb_devices = int(os.environ.get('MS_WORKER_NUM', -1))
        rt_devices = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", None)
        if rt_devices is not None:
            rt_devices = [int(i) for i in rt_devices.split(',')]
            assert (self.nb_devices <= len(rt_devices))
            if self.nb_devices < len(rt_devices):
                rt_devices = rt_devices[:self.nb_devices]
        elif self.nb_devices != -1:
            rt_devices = list(range(self.nb_devices))
        else:
            rt_devices = list(range(8))
        assert rt_devices is not None

        self.lock_prefill_launch = posix_ipc.Semaphore(
            name="/lock_prefill_launch", flags=0)
        self.lock_prefill_launch_devices = []
        for i in rt_devices:
            self.lock_prefill_launch_devices.append(
                posix_ipc.Semaphore(name=f"/lock_prefill_launch{i}", flags=0))
        self.lock_decode_launch = posix_ipc.Semaphore(
            name="/lock_decode_launch", flags=0)
        self.lock_decode_launch_devices = []
        for i in rt_devices:
            self.lock_decode_launch_devices.append(
                posix_ipc.Semaphore(name=f"/lock_decode_launch{i}", flags=0))
        self.lock_logits_launch = posix_ipc.Semaphore(
            name="/lock_logits_launch", flags=0)
        self.lock_logits_launch_devices = []
        for i in rt_devices:
            self.lock_logits_launch_devices.append(
                posix_ipc.Semaphore(name=f"/lock_logits_launch{i}", flags=0))

        self.id = os.environ.get("MASTER_PORT", 0)
        if self.id == 0:
            logger.warning("please set a MASTER_PORT,  export MASTER_PORT=XXX")

        self.rw = _init_semaphore(name=f"/{self.id}.rw", initial_value=1)
        self.barrier_ready = _init_semaphore(name=f"/{self.id}.barrier_ready",
                                             initial_value=0)
        (self.counter_memory_ready,
         self.counter_file_ready) = _init_shared_memory(
             name=f"/{self.id}.counter_ready", size=1, initial_value=0)
        self.barrier_done = _init_semaphore(name=f"/{self.id}.barrier_done",
                                            initial_value=0)
        (self.counter_memory_done,
         self.counter_file_done) = _init_shared_memory(
             name=f"/{self.id}.counter_done", size=1, initial_value=0)

        self.need_release_logits = False

    def __del__(self):
        self._cleanup_ipc()

    def __exit__(self):
        self._cleanup_ipc()

    def _cleanup_ipc(self):
        # clean global shared memory
        self.lock_prefill_launch.close()
        for sem in self.lock_prefill_launch_devices:
            sem.close()
        self.lock_decode_launch.close()
        for sem in self.lock_decode_launch_devices:
            sem.close()
        self.lock_logits_launch.close()
        for sem in self.lock_logits_launch_devices:
            sem.close()

        # clean local LLM shared memory generated
        self.rw.close()
        self.barrier_ready.close()
        self.counter_file_ready.close()
        self.barrier_done.close()
        self.counter_file_done.close()
        with suppress(posix_ipc.ExistentialError):
            self.rw.unlink()
        with suppress(posix_ipc.ExistentialError):
            self.barrier_ready.unlink()
        with suppress(posix_ipc.ExistentialError):
            self.counter_memory_ready.unlink()
        with suppress(posix_ipc.ExistentialError):
            self.barrier_done.unlink()
        with suppress(posix_ipc.ExistentialError):
            self.counter_memory_done.unlink()

    def _update_semaphore_list(self):
        if len(self.lock_prefill_launch_devices) != self.nb_devices:
            rt_devices = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", None)
            if rt_devices is not None:
                rt_devices = [int(i) for i in rt_devices.split(',')]
                assert (self.nb_devices <= len(rt_devices))
                if self.nb_devices < len(rt_devices):
                    rt_devices = rt_devices[:self.nb_devices]
            else:
                rt_devices = list(range(self.nb_devices))

            for lock_prefill_launch_device in self.lock_prefill_launch_devices:
                lock_prefill_launch_device.close()
            self.lock_prefill_launch_devices.clear()
            for lock_decode_launch_device in self.lock_decode_launch_devices:
                lock_decode_launch_device.close()
            self.lock_decode_launch_devices.clear()
            for lock_logits_launch_device in self.lock_logits_launch_devices:
                lock_logits_launch_device.close()
            self.lock_logits_launch_devices.clear()

            for i in rt_devices:
                self.lock_prefill_launch_devices.append(
                    posix_ipc.Semaphore(name=f"/lock_prefill_launch{i}",
                                        flags=0))
                self.lock_decode_launch_devices.append(
                    posix_ipc.Semaphore(name=f"/lock_decode_launch{i}",
                                        flags=0))
                self.lock_logits_launch_devices.append(
                    posix_ipc.Semaphore(name=f"/lock_logits_launch{i}",
                                        flags=0))

    def check_number_devices(self):
        if self.nb_devices == -1:
            self.nb_devices = int(os.environ.get('MS_WORKER_NUM', -1))
            assert (self.nb_devices != -1)
            self._update_semaphore_list()

    def prefill_ready(self, timeout: Optional[float] = None):
        '''
        Check that the LLM is ready to launch prefill,
        if so return True else return False
        Can be call before launching the prefill instead of wait_prefill_ready

        Acquire the prefill locks

        Guarantee all process is ready
        Check that no other LLM (using this lib) 
        is currently launching a prefill

        NB: (not implemented yet) can add a timeout, in this case return False

        @param timeout(None|float) Optional: time to wait to acquire the lock 
        (not implemented yet)
        @return True if prefill can be launched, else return False

        @see wait_prefill_ready
        '''
        self.rw.acquire()

        self.counter_file_ready.seek(0)
        ctn = self.counter_file_ready.read_byte()
        ctn += 1

        if ctn == self.nb_devices:
            # Need if we consider that
            # the device id are not order and consecutive
            self.lock_prefill_launch.acquire()
            for lock_prefill_launch_device in self.lock_prefill_launch_devices:
                lock_prefill_launch_device.acquire()

            self.counter_file_ready.seek(0)
            self.counter_file_ready.write_byte(0)
            for _ in range(self.nb_devices):
                self.barrier_ready.release()
            self.lock_prefill_launch.release()
        else:
            self.counter_file_ready.seek(0)
            self.counter_file_ready.write_byte(ctn)

        self.rw.release()
        self.barrier_ready.acquire()
        return True

    def wait_prefill_ready(self,
                           period: float = 0.001,
                           timeout: Optional[float] = None):
        '''
        Call before launching the prefill

        Wait until the prefill is ready to be launched

        @param period(float): sleep period between call prefill_ready
        @param timeout(None|float) Optional:
        time to wait to acquire the lock (not implemented yet)

        @see prefill_ready
        '''
        while (not self.prefill_ready()):
            time.sleep(period)

    def prefill_done(self):
        '''
        Call when the prefill is finish to run

        Release prefill lock
        '''
        self.rw.acquire()

        self.counter_file_ready.seek(0)
        ctn = self.counter_file_ready.read_byte()
        ctn += 1

        if ctn == self.nb_devices:
            # self.lock_prefill_launch.acquire()
            # Need if we consider that
            # the device id are not order and consecutive
            for lock_prefill_launch_device in self.lock_prefill_launch_devices:
                lock_prefill_launch_device.release()

            self.counter_file_ready.seek(0)
            self.counter_file_ready.write_byte(0)
            for _ in range(self.nb_devices):
                self.barrier_ready.release()
            # self.lock_prefill_launch.release()
        else:
            self.counter_file_ready.seek(0)
            self.counter_file_ready.write_byte(ctn)

        self.rw.release()
        self.barrier_ready.acquire()

    def decode_ready(self, timeout: Optional[float] = None):
        '''
        Check that the LLM is ready to launch decode,
        if so return True else return False
        Can be call before launching the decode instead of wait_decode_ready

        @param timeout(None|float) Optional: time to wait to acquire the lock
        (not implemented yet)
        @return True if decode can be launched, else return False

        @see wait_decode_ready
        '''
        return True

    def wait_decode_ready(self,
                          period: float = 0.001,
                          timeout: Optional[float] = None):
        '''
        Call before launching the decode

        Wait until the decode is ready to be launched

        @param period(float): sleep period between call decode_ready
        @param timeout(None|float) Optional: time to wait to acquire the lock
        (not implemented yet)

        @see decode_ready
        '''
        while (not self.decode_ready()):
            time.sleep(period)

    def decode_done(self):
        '''
        Call when the decode is finish to run
        '''
        pass

    def logits_ready(self, timeout: Optional[float] = None):
        '''
        Check that the LLM is ready to launch logits,
        if so return True else return False
        Can be call before launching the logits instead of wait_logits_ready

        Acquire the logits locks

        Guarantee all process is ready
        Check that no other LLM (using this lib) is currently launching a logits

        NB: (not implemented yet) can add a timeout, in this case return False

        @param timeout(None|float) Optional: time to wait to acquire the lock
        (not implemented yet)
        @return True if logits can be launched, else return False

        @see wait_logits_ready
        '''
        if (self.lock_prefill_launch_devices[0].value != 0
                or self.lock_decode_launch_devices[0].value != 0):
            self.rw.acquire()

            self.counter_file_ready.seek(0)
            ctn = self.counter_file_ready.read_byte()
            ctn += 1

            if ctn == self.nb_devices:
                # Need if we consider that
                # the device id are not order and consecutive
                self.lock_logits_launch.acquire()
                for lock_logits_launch_device in (
                        self.lock_logits_launch_devices):
                    lock_logits_launch_device.acquire()

                self.counter_file_ready.seek(0)
                self.counter_file_ready.write_byte(0)
                for _ in range(self.nb_devices):
                    self.barrier_ready.release()
                self.lock_logits_launch.release()
            else:
                self.counter_file_ready.seek(0)
                self.counter_file_ready.write_byte(ctn)

            self.rw.release()
            self.barrier_ready.acquire()
            self.need_release_logits = True
        else:
            self.need_release_logits = False
        return True

    def wait_logits_ready(self,
                          period: float = 0.001,
                          timeout: Optional[float] = None):
        '''
        Call before launching the logits

        Wait until the logits is ready to be launched

        @param period(float): sleep period between call logits_ready
        @param timeout(None|float) Optional: time to wait to acquire the lock
        (not implemented yet)

        @see logits_ready
        '''
        while (not self.logits_ready()):
            time.sleep(period)

    def logits_done(self):
        '''
        Call when the logits is finish to run

        Release logits lock
        '''
        if self.need_release_logits:
            self.rw.acquire()

            self.counter_file_ready.seek(0)
            ctn = self.counter_file_ready.read_byte()
            ctn += 1

            if ctn == self.nb_devices:
                # self.lock_logits_launch.acquire()
                # Need if we consider that
                # the device id are not order and consecutive
                for lock_logits_launch_device in (
                        self.lock_logits_launch_devices):
                    lock_logits_launch_device.release()

                self.counter_file_ready.seek(0)
                self.counter_file_ready.write_byte(0)
                for _ in range(self.nb_devices):
                    self.barrier_ready.release()
                # self.lock_logits_launch.release()
            else:
                self.counter_file_ready.seek(0)
                self.counter_file_ready.write_byte(ctn)

            self.rw.release()
            self.barrier_ready.acquire()


class CollocatorMaster:
    '''Class to collocate execution through IPC
    When a master process manage:
    the synchronization the launch of prefill/decode for all the devices
    '''

    def __init__(self):
        logger.info("CollocatorMaster: __init__")
        if os.environ.get("MS_ENABLE_LCCL", False) not in ['on']:
            logger.warning("May need to export MS_ENABLE_LCCL=on")
        ms_dev_runtime_conf = os.environ.get("MS_DEV_RUNTIME_CONF", False)
        if (ms_dev_runtime_conf is False
                or "comm_init_lccl_only:true" not in ms_dev_runtime_conf):
            logger.warning("May need to export "
                           "MS_DEV_RUNTIME_CONF=\"comm_init_lccl_only:true\"")
        disable_custom_kernel = os.environ.get(
            "MS_INTERNAL_DISABLE_CUSTOM_KERNEL_LIST", False)
        if (disable_custom_kernel is False
                or "MatMulAllReduce" not in disable_custom_kernel):
            logger.warning(
                "May need to export "
                "MS_INTERNAL_DISABLE_CUSTOM_KERNEL_LIST=MatMulAllReduce")

        rt_devices = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", None)
        if rt_devices is not None:
            rt_devices = [int(i) for i in rt_devices.split(',')]
        else:
            nb_devices = int(os.environ.get('MS_WORKER_NUM'))
            rt_devices = list(range(nb_devices))
        assert rt_devices is not None

        self.lock_prefill_launch = posix_ipc.Semaphore(
            name="/lock_prefill_launch", flags=0)
        self.lock_prefill_launch_devices = []
        for i in rt_devices:
            self.lock_prefill_launch_devices.append(
                posix_ipc.Semaphore(name=f"/lock_prefill_launch{i}", flags=0))
        self.lock_decode_launch = posix_ipc.Semaphore(
            name="/lock_decode_launch", flags=0)
        self.lock_decode_launch_devices = []
        for i in rt_devices:
            self.lock_decode_launch_devices.append(
                posix_ipc.Semaphore(name=f"/lock_decode_launch{i}", flags=0))
        self.lock_logits_launch = posix_ipc.Semaphore(
            name="/lock_logits_launch", flags=0)
        self.lock_logits_launch_devices = []
        for i in rt_devices:
            self.lock_logits_launch_devices.append(
                posix_ipc.Semaphore(name=f"/lock_logits_launch{i}", flags=0))

    def __del__(self):
        self._cleanup_ipc()

    def __exit__(self):
        self._cleanup_ipc()

    def _cleanup_ipc(self):
        self.lock_prefill_launch.close()
        for lock_prefill_launch_device in self.lock_prefill_launch_devices:
            lock_prefill_launch_device.close()
        self.lock_decode_launch.close()
        for lock_decode_launch_device in self.lock_decode_launch_devices:
            lock_decode_launch_device.close()
        self.lock_logits_launch.close()
        for lock_logits_launch_device in self.lock_logits_launch_devices:
            lock_logits_launch_device.close()

    def prefill_ready(self, timeout: Optional[float] = None):
        '''
        Check that the LLM is ready to launch prefill,
        if so return True else return False
        Can be call before launching the prefill instead of wait_prefill_ready

        Acquire the prefill locks

        Check that no other LLM (using this lib)
        is currently launching a prefill

        NB: (not implemented yet) can add a timeout, in this case return False

        @param timeout(None|float) Optional: time to wait to acquire the lock
        (not implemented yet)
        @return True if prefill can be launched, else return False

        @see wait_prefill_ready
        '''
        # Need if we consider that the device id are not order and consecutive
        self.lock_prefill_launch.acquire()
        for lock_prefill_launch_device in self.lock_prefill_launch_devices:
            lock_prefill_launch_device.acquire()
        self.lock_prefill_launch.release()
        return True

    def wait_prefill_ready(self,
                           period: float = 0.001,
                           timeout: Optional[float] = None):
        '''
        Call before launching the prefill

        Wait until the prefill is ready to be launched

        @param period(float): sleep period between call prefill_ready
        @param timeout(None|float) Optional: time to wait to acquire the lock
        (not implemented yet)

        @see prefill_ready
        '''
        while (not self.prefill_ready()):
            time.sleep(period)

    def prefill_done(self):
        '''
        Call when the prefill is finish to run

        Release prefill lock
        '''
        # self.lock_prefill_launch.acquire()
        for lock_prefill_launch_device in self.lock_prefill_launch_devices:
            lock_prefill_launch_device.release()
        # self.lock_prefill_launch.release()

    def decode_ready(self, timeout: Optional[float] = None):
        '''
        Check that the LLM is ready to launch decode,
        if so return True else return False
        Can be call before launching the decode instead of wait_decode_ready

        @param timeout(None|float) Optional: time to wait to acquire the lock
        (not implemented yet)
        @return True if decode can be launched, else return False

        @see wait_decode_ready
        '''
        return True

    def wait_decode_ready(self,
                          period: float = 0.001,
                          timeout: Optional[float] = None):
        '''
        Call before launching the decode

        Wait until the decode is ready to be launched

        @param period(float): sleep period between call decode_ready
        @param timeout(None|float) Optional: time to wait to acquire the lock
        (not implemented yet)

        @see decode_ready
        '''
        while (not self.decode_ready()):
            time.sleep(period)

    def decode_done(self):
        '''
        Call when the decode is finish to run
        '''
        pass

    def logits_ready(self, timeout: Optional[float] = None):
        '''
        Check that the LLM is ready to launch logits,
        if so return True else return False
        Can be call before launching the logits instead of wait_logits_ready

        Acquire the logits locks

        Check that no other LLM (using this lib) is currently launching a logits

        NB: (not implemented yet) can add a timeout, in this case return False

        @param timeout(None|float) Optional: time to wait to acquire the lock
        (not implemented yet)
        @return True if logits can be launched, else return False

        @see wait_logits_ready
        '''
        # Need if we consider that the device id are not order and consecutive
        self.lock_logits_launch.acquire()
        for lock_logits_launch_device in self.lock_logits_launch_devices:
            lock_logits_launch_device.acquire()
        self.lock_logits_launch.release()
        return True

    def wait_logits_ready(self,
                          period: float = 0.001,
                          timeout: Optional[float] = None):
        '''
        Call before launching the logits

        Wait until the logits is ready to be launched

        @param period(float): sleep period between call logits_ready
        @param timeout(None|float) Optional: time to wait to acquire the lock
        (not implemented yet)

        @see logits_ready
        '''
        while (not self.logits_ready()):
            time.sleep(period)

    def logits_done(self):
        '''
        Call when the logits is finish to run

        Release logits lock
        '''
        # self.lock_logits_launch.acquire()
        for lock_logits_launch_device in self.lock_logits_launch_devices:
            lock_logits_launch_device.release()
        # self.lock_logits_launch.release()
