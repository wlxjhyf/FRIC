import os
import ctypes


file_name = "/home/panda/xujiahao/fric_riscv/kv_odirect"

# 文件打开
fd = os.open(file_name, os.O_WRONLY | os.O_CREAT | os.O_DIRECT, 0o666)

# 分配页对齐缓冲区（假设 4KB 对齐）
alignment = 4096
size = 4096  # 必须是对齐大小的整数倍
buf = ctypes.create_string_buffer(size + alignment)
addr = ctypes.addressof(buf)
offset = (alignment - (addr % alignment)) % alignment # 到下一个对齐地址的偏移量
aligned_buf = (ctypes.c_char * size).from_address(addr + offset)

# 填充数据
# 这里可以通过 numpy 或 ctypes memcpy 把 Torch Tensor 数据拷到 aligned_buf
import torch, numpy as np
tensor = torch.randn(size // 4, dtype=torch.float32)  # 4 bytes per float
np_array = np.frombuffer(aligned_buf, dtype=np.float32, count=tensor.numel())
np_array[:] = tensor.cpu().numpy()

# 写入 O_DIRECT
os.write(fd, aligned_buf)
os.close(fd)