import os
import torch
import ctypes
from liburing import (
    io_uring, io_uring_queue_init, io_uring_get_sqe, io_uring_prep_write,
    io_uring_submit, io_uring_wait_cqe, io_uring_cqe, io_uring_cqe_seen, iovec,
    io_uring_queue_exit
)


'''
将FRIC的提前分配k_buf与v_buf的逻辑与这里结合,
直接实现将buf注册为对齐的内存
假设每次写都是一个B、T、head_dim、num_heads大小的数据, KV两个数据
'''
# 先实现提前分配好，能够容纳最大 tokens （也就是支持prefill）的构建buf

def buffer_prepare(max_batch_size, num_heads, max_seq_len, head_dim, dtype):
    k_buf = torch.empty((max_batch_size, num_heads, max_seq_len, head_dim), device='cpu', dtype=dtype, pin_memory=True)
    v_buf = torch.empty((max_batch_size, num_heads, max_seq_len, head_dim), device='cpu', dtype=dtype, pin_memory=True)

    return k_buf, v_buf






