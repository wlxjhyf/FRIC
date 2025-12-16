import os
import ctypes
import numpy as np
from collections import deque
import threading
import fcntl
from .index_table import FRIC_SSD_Log

BLOCK_SIZE = 2 * 1024 * 1024  # 2MB
BLOCK_COUNT = 1000  # 100 / 2 * 1024
PATH = "/home/panda/xujiahao/fric_riscv"


class BlockPool:
    def __init__(self, fds):
        """
        fds: List[int]
        每个 fd 对应一个物理 block（文件）
        """
        # 保存完整的物理块：(block_index, fd)
        self._free = deque((i, fd) for i, fd in enumerate(fds))
        self._lock = threading.Lock()
        self.capacity = len(fds)

    def _try_acquire(self, fd) -> bool:
        """
        尝试跨进程独占这个 block
        """
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except BlockingIOError:
            return False

    def _release(self, fd):
        fcntl.flock(fd, fcntl.LOCK_UN)

    def alloc(self):
        """
        分配一个物理 block
        return: (block_index, fd) or None
        """
        with self._lock:
            n = len(self._free)
            for _ in range(n):
                block = self._free.popleft()
                idx, fd = block

                if self._try_acquire(fd):
                    return block
                else:
                    # 被别的进程占用，放回队尾
                    self._free.append(block)

            return None

    def free(self, block):
        """
        归还一个物理 block
        block: (block_index, fd)
        """
        idx, fd = block
        self._release(fd)

        with self._lock:
            self._free.append(block)

    def available(self):
        with self._lock:
            return len(self._free)

class BlockBinding:
    def __init__(self, block_pool: BlockPool):
        """
        vblock -> pblock 映射
        pblock = (block_index, fd)
        """
        self.v2p = []          # index = vblock
        self.block_pool = block_pool
        self._lock = threading.Lock()

    def alloc(self) -> int:
        """
        分配一个新的虚拟块号
        return: vblock
        """
        block = self.block_pool.alloc()
        if block is None:
            raise RuntimeError("FRIC: No free physical block available")

        with self._lock:
            vblock = len(self.v2p)
            self.v2p.append(block)
            return vblock

    def lookup(self, vblock: int):
        """
        return: (block_index, fd)
        """
        block = self.v2p[vblock]
        if block is None:
            raise RuntimeError(f"vblock {vblock} already freed")
        return block

    def fd_of(self, vblock: int) -> int:
        return self.lookup(vblock)[1]

    def free(self, vblock: int):
        """
        释放一个虚拟块
        """
        with self._lock:
            block = self.v2p[vblock]
            if block is None:
                return
            self.v2p[vblock] = None

        self.block_pool.free(block)



def blocks_prepare():
    fds = []
    for i in range(BLOCK_COUNT):
        filename = os.path.join(PATH, f"block_{i:05d}")
        fd = os.open(filename, os.O_CREAT | os.O_RDWR | os.O_DIRECT, 0o666)
        # os.posix_fallocate(fd, 0, BLOCK_SIZE)
        fds.append(fd)
    return BlockPool(fds)


def log_prepare():
    fric_ssd_log = FRIC_SSD_Log()
    return fric_ssd_log


if __name__ == "__main__":
    blocks_prepare()
    log_prepare()