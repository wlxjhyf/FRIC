import os
import ctypes
import numpy as np

BLOCK_SIZE = 2 * 1024 * 1024  # 2MB
BLOCK_COUNT = 1000  # 100 / 2 * 1024
PATH = "/home/panda/xujiahao/fric_riscv"

def blocks_prepare():
    fds = []
    for i in range(BLOCK_COUNT):
        filename = os.path.join(PATH, f"block_{i:05d}")
        fd = os.open(filename, os.O_CREAT | os.O_RDWR | os.O_DIRECT, 0o666)
        os.posix_fallocate(fd, 0, BLOCK_SIZE)
        fds.append(fd)
    return fds


if __name__ == "__main__":
    blocks_prepare()