import os
import torch
import ctypes

from liburing import (
    io_uring, io_uring_queue_init, io_uring_get_sqe, io_uring_prep_write,
    io_uring_submit, io_uring_wait_cqe, io_uring_cqe, io_uring_cqe_seen, iovec,
    io_uring_queue_exit
)

BLOCK_COUNT = 16
BLOCK_SIZE = 4096  # bytes
PATH = "/home/panda/xujiahao/fric_riscv"

# 准备 io_uring
ring = io_uring()
cqe = io_uring_cqe()
io_uring_queue_init(BLOCK_COUNT, ring, 0)

# 分配 CPU pinned memory tensor
blocks = []
fds = []
for i in range(BLOCK_COUNT):
    filename = os.path.join(PATH, f"block_{i:05d}")
    fd = os.open(filename, os.O_CREAT | os.O_RDWR | os.O_DIRECT, 0o666)
    tensor = torch.empty(BLOCK_SIZE // 4, dtype=torch.float32, device='cpu', pin_memory=True)
    blocks.append(tensor)
    fds.append(fd)

# 提交写入函数
def submit_writes_pytorch(ring, cqe, blocks, fds):
    for buf, fd in zip(blocks, fds):
        sqe = io_uring_get_sqe(ring)
        ptr = ctypes.c_void_p(buf.data_ptr())        # tensor 内存地址
        length = buf.numel() * buf.element_size()   # bytes
        iov = iovec(memoryview((ctypes.c_char * length).from_address(ptr.value)))
        io_uring_prep_write(sqe, fd, iov.iov_base, iov.iov_len, 0)
    io_uring_submit(ring)
    for _ in blocks:
        io_uring_wait_cqe(ring, cqe)
        io_uring_cqe_seen(ring, cqe)

# 写入数据
submit_writes_pytorch(ring, cqe, blocks, fds)

# 关闭文件和退出 io_uring
for fd in fds:
    os.close(fd)
io_uring_queue_exit(ring)