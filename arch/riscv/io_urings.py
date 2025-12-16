import os
import ctypes
import torch
import numpy as np
import faulthandler
faulthandler.enable()
import threading

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
from .index_table import KVIndexTable, IOEntry
from .ssd_blocks import blocks_prepare

# ENTRIES = 64 
# kv_index_table = KVIndexTable()


# ring = io_uring()
# cqe = io_uring_cqe()
# io_uring_queue_init(ENTRIES, ring, 0) # Flag可选：IORING_SETUP_SQPOLL


# def cqe_worker(ring, fric_ssd_log):
#     cqe = io_uring_cqe()

#     while True:
#         ret = io_uring_wait_cqe(ring, cqe)
#         if ret < 0:
#             continue

#         io_entry = ctypes.cast(
#             ctypes.c_void_p(cqe.user_data),
#             ctypes.py_object
#         ).value

#         io_entry.pending -= 1

#         if io_entry.pending == 0:
#             io_entry.done = True

#             kv_index_table.commit_entry(
#                 io_entry.seq,
#                 io_entry.kv,
#                 io_entry.layer,
#                 io_entry.token,
#                 io_entry.start_idx,
#                 io_entry.end_idx,
#                 io_entry.start_offset,
#                 io_entry.end_offset,
#             )

#         io_uring_cqe_seen(ring, cqe)
        
#         if io_entry.end_idx == io_entry.start_idx:
#             fric_ssd_log.log_partial(io_entry.start_idx, io_entry.end_offset - io_entry.start_offset)
#         elif io_entry.end_idx == io_entry.start_idx + 1:
#             fric_ssd_log.log_partial(io_entry.start_idx, io_entry.end_idx)
#         else:
#             RuntimeError("FRIC: the token level multi write is TODO")


# threading.Thread(
#     target=cqe_worker,
#     args=(ring,),
#     daemon=True
# ).start()


# def __write_iov(ring, block_binding, iov, start_idx, start_offset, capacity, io_entry):
#     base = iov.iov_base
#     length = iov.iov_len

#     work_idx = start_idx
#     work_offset = start_offset

#     written = 0

#     if work_idx == 0 and work_offset == 0:
#         _ = block_binding.alloc()

#     while written < length:
#         sqe = io_uring_get_sqe(ring)

#         now_capacity = capacity - work_offset
#         need = length - written
#         write_len = min(now_capacity, need)

#         chunk = base[written : written + write_len]

#         fd = block_binding.fd_of(work_idx)

#         io_uring_prep_write(
#             sqe,
#             fd,
#             chunk,
#             write_len,
#             work_offset
#         )

#         sqe.user_data = ctypes.cast(
#             ctypes.py_object(io_entry),
#             ctypes.c_void_p
#         ).value

#         io_entry.pending += 1
        
#         written += write_len
#         work_offset += write_len

#         if work_offset == capacity:
#             work_idx += 1
#             _ = block_binding.alloc()
#             work_offset = 0

#     io_uring_submit(ring)

#     return work_idx, work_offset

def make_iov(buf):
    ptr = ctypes.c_void_p(buf.contiguous().data_ptr())
    length = buf.numel() * buf.element_size()
    iov = iovec(memoryview((ctypes.c_char * length).from_address(ptr.value)))
    return iov


# # token不是从0开始的
# def submit(block_binding, k_buf, v_buf, seq, layer, token):
#     # print(f"k_buf:{k_buf.shape} v_buf:{v_buf.shape} layer:{layer} token:{token}")
#     k_iov = __make_iov(k_buf)
#     v_iov = __make_iov(v_buf)
#     for kv, iov in (("K", k_iov), ("V", v_iov)):
#         start_idx = kv_index_table.idx
#         start_offset = kv_index_table.offset

#         io_entry = IOEntry(
#             seq=seq,
#             kv=kv,
#             layer=layer,
#             token=token,
#             start_idx=start_idx,
#             start_offset=start_offset,
#         )

#         end_idx, end_offset = __write_iov(
#             ring,
#             block_binding,
#             iov,
#             start_idx,
#             start_offset,
#             kv_index_table.capacity,
#             io_entry
#         )

#         io_entry.end_idx = end_idx
#         io_entry.end_offset = end_offset

#         kv_index_table.add_entry(
#             seq, kv, layer, token,
#             start_idx, end_idx,
#             start_offset, end_offset
#         )

