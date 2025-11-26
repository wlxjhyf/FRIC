import os
import ctypes
import numpy as np
import torch
from liburing import (
    io_uring,
    io_uring_queue_init,
    io_uring_get_sqe,
    io_uring_prep_write,
    io_uring_submit,
    io_uring_wait_cqe,
    io_uring_cqe_seen,
    io_uring_queue_exit,
    io_uring_cqe,
    iovec,
)

BLOCK_SIZE = 2 * 1024 * 1024  # 2MB
BLOCK_COUNT = 10  # 100 / 2 * 1024
PATH = "/home/panda/xujiahao/fric_riscv"
ALIGNMENT = 4096  # 4KB 对齐


def aligned_buffer(size, alignment=ALIGNMENT):
    buf = ctypes.create_string_buffer(size + alignment)
    addr = ctypes.addressof(buf)
    offset = (alignment - (addr % alignment)) % alignment
    return (ctypes.c_char * size).from_address(addr + offset)

def prepare_blocks():
    blocks = []
    for i in range(BLOCK_COUNT):
        filename = os.path.join(PATH, f"block_{i:05d}")
        fd = os.open(filename, os.O_CREAT | os.O_RDWR | os.O_DIRECT, 0o666)
        buf = aligned_buffer(BLOCK_SIZE)
        blocks.append({"fd": fd, "buf": buf})
    return blocks

def submit_writes(ring, cqe, blocks):
    for block in blocks:
        sqe = io_uring_get_sqe(ring)
        iov = iovec(memoryview(block["buf"]))
        addr = iov.iov_base
        length = iov.iov_len
        io_uring_prep_write(sqe, block["fd"], addr, length, 0)
    io_uring_submit(ring)

    for _ in blocks:
        io_uring_wait_cqe(ring, cqe)
        io_uring_cqe_seen(ring, cqe)

def main():
    os.makedirs(PATH, exist_ok=True)

    ring = io_uring()
    cqe = io_uring_cqe()
    io_uring_queue_init(8, ring, 0)

    blocks = prepare_blocks()
    # submit_writes(ring, cqe, blocks)

    print(f"{len(blocks)} blocks written and open.")

    io_uring_queue_exit(ring)

if __name__ == "__main__":
    main()