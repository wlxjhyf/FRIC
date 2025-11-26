import os
import ctypes
import torch
import numpy as np
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

# 准备 io_uring
ring = io_uring()
cqe = io_uring_cqe()
# 队列大小可以调，根据并发 I/O 数量
io_uring_queue_init(8, ring, 0)

# 打开文件（O_DIRECT）
file_name = "/home/panda/xujiahao/fric_riscv/kv_odirect"
fd = os.open(file_name, os.O_WRONLY | os.O_CREAT | os.O_DIRECT, 0o666)

# 分配对齐缓冲区
alignment = 4096
size = 4096
buf = ctypes.create_string_buffer(size + alignment)
addr = ctypes.addressof(buf)
offset = (alignment - (addr % alignment)) % alignment
aligned_buf = (ctypes.c_char * size).from_address(addr + offset)

# 填充数据（从 GPU Tensor 拷贝到这个 aligned_buf）
tensor = torch.randn(size // 4, dtype=torch.float32)
np_arr = np.frombuffer(aligned_buf, dtype=np.float32, count=tensor.numel())
np_arr[:] = tensor.cpu().numpy()

# 用 io_uring 提交异步写
sqe = io_uring_get_sqe(ring)

# 提交
# addr = bytearray(aligned_buf)
# length = size
iov = iovec(memoryview(aligned_buf))
addr = iov.iov_base
length = iov.iov_len

io_uring_prep_write(sqe, fd, addr, length, 0)
io_uring_submit(ring)

# 等待完成
io_uring_wait_cqe(ring, cqe)
res = cqe.res  # 写入返回值 (bytes 写了多少，或错误)
if res < 0:
    raise IOError(f"io_uring write failed: {res}")
# 告诉 io_uring 我们处理了这个完成队列项
io_uring_cqe_seen(ring, cqe)

# 释放 / 关闭
os.close(fd)
# 退出 io_uring
io_uring_queue_exit(ring)


#### example ####
# from liburing import O_CREAT, O_RDWR, AT_FDCWD, iovec, io_uring, io_uring_get_sqe, \
#                      io_uring_prep_openat, io_uring_prep_write, io_uring_prep_read, \
#                      io_uring_prep_close, io_uring_submit, io_uring_wait_cqe, \
#                      io_uring_cqe_seen, io_uring_cqe, io_uring_queue_init, io_uring_queue_exit, \
#                      io_uring_sqe_set_data64, trap_error


# def open(ring, cqe, path, flags, mode=0o660, dir_fd=AT_FDCWD):
#     _path = path if isinstance(path, bytes) else str(path).encode()
#     # if `path` is relative and `dir_fd` is `AT_FDCWD`, then `path` is relative
#     # to current working directory. Also `_path` must be in bytes

#     sqe = io_uring_get_sqe(ring)  # sqe(submission queue entry)
#     io_uring_prep_openat(sqe, _path, flags, mode, dir_fd)
#     # set submit entry identifier as `1` which is returned back in `cqe.user_data`
#     # so you can keep track of submit/completed entries.
#     io_uring_sqe_set_data64(sqe, 1)
#     return _submit_and_wait(ring, cqe)  # returns fd


# def write(ring, cqe, fd, data, offset=0):
#     iov = iovec(data)  # or iovec([bytearray(data)])
#     sqe = io_uring_get_sqe(ring)
#     io_uring_prep_write(sqe, fd, iov.iov_base, iov.iov_len, offset)
#     io_uring_sqe_set_data64(sqe, 2)
#     return _submit_and_wait(ring, cqe)  # returns length(s) of bytes written


# def read(ring, cqe, fd, length, offset=0):
#     iov = iovec(bytearray(length))  # or [bytearray(length)]
#     sqe = io_uring_get_sqe(ring)
#     io_uring_prep_read(sqe, fd, iov.iov_base, iov.iov_len, offset)
#     io_uring_sqe_set_data64(sqe, 3)
#     _submit_and_wait(ring, cqe)  # get actual length of file read.
#     return iov.iov_base


# def close(ring, cqe, fd):
#     sqe = io_uring_get_sqe(ring)
#     io_uring_prep_close(sqe, fd)
#     io_uring_sqe_set_data64(sqe, 4)
#     _submit_and_wait(ring, cqe)  # no error means success!


# def _submit_and_wait(ring, cqe):
#     io_uring_submit(ring)  # submit entry
#     io_uring_wait_cqe(ring, cqe)  # wait for entry to finish
#     result = trap_error(cqe.res)  # auto raise appropriate exception if failed
#     # note `cqe.res` returns results, if ``< 0`` its an error, if ``>= 0`` its the value

#     # done with current entry so clear it from completion queue.
#     io_uring_cqe_seen(ring, cqe)
#     return result  # type: int


# def main():
#     ring = io_uring()
#     cqe = io_uring_cqe()  # completion queue entry
#     try:
#         io_uring_queue_init(32, ring, 0)

#         fd = open(ring, cqe, '/tmp/liburing-test-file.txt', O_CREAT | O_RDWR)
#         print('fd:', fd)

#         length = write(ring, cqe, fd, b'hello world')
#         print('wrote:', length)

#         content = read(ring, cqe, fd, length)
#         print('read:', content)

#         close(ring, cqe, fd)
#         print('closed.')
#     finally:
#         io_uring_queue_exit(ring)


# if __name__ == '__main__':
#     main()