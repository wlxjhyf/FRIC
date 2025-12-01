import os
import ctypes
import torch
import numpy as np
import faulthandler
faulthandler.enable()

from liburing import (
    io_uring,
    io_uring_queue_init,
    io_uring_get_sqe,
    io_uring_prep_write,
    io_uring_submit,
    io_uring_wait_cqe,
    io_uring_cqe,
    io_uring_cqe_seen,
    iovec,
    io_uring_queue_exit,
)
from .index_table import KVIndexTable

ENTRIES = 64 
kv_index_table = KVIndexTable()

ring = io_uring()
cqe = io_uring_cqe()
io_uring_queue_init(ENTRIES, ring, 0) # Flag可选：IORING_SETUP_SQPOLL


def __write_iov(ring, fds, iov, start_idx, start_offset, capacity):
    base = iov.iov_base
    length = iov.iov_len

    work_idx = start_idx
    work_offset = start_offset

    written = 0

    while written < length:
        sqe = io_uring_get_sqe(ring)

        now_capacity = capacity - work_offset
        need = length - written
        write_len = min(now_capacity, need)

        chunk = base[written : written + write_len]

        io_uring_prep_write(
            sqe,
            fds[work_idx],
            chunk,
            write_len,
            work_offset
        )

        written += write_len
        work_offset += write_len

        if work_offset == capacity:
            work_idx += 1
            work_offset = 0

    io_uring_submit(ring)
    # io_uring_wait_cqe(ring, cqe)
    # io_uring_cqe_seen(ring, cqe)

    return work_idx, work_offset

def __make_iov(buf):
    ptr = ctypes.c_void_p(buf.contiguous().data_ptr())
    length = buf.numel() * buf.element_size()
    iov = iovec(memoryview((ctypes.c_char * length).from_address(ptr.value)))
    return iov


# token不是从0开始的
def submit(fds, k_buf, v_buf, seq, layer, token):
    # print(f"k_buf:{k_buf.shape} v_buf:{v_buf.shape} layer:{layer} token:{token}")
    k_iov = __make_iov(k_buf)
    v_iov = __make_iov(v_buf)
    for kv, iov in (("K", k_iov), ("V", v_iov)):
        start_idx = kv_index_table.idx
        start_offset = kv_index_table.offset

        end_idx, end_offset = __write_iov(
            ring,
            fds,
            iov,
            start_idx,
            start_offset,
            kv_index_table.capacity
        )

        kv_index_table.add_entry(
            seq, kv, layer, token,
            start_idx, end_idx,
            start_offset, end_offset
        )

