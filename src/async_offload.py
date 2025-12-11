import torch
import os
import mmap
import threading
import numpy as np
import cupy as cp

'''
PURE_INF: only inference without kv offload
USER_SPACE: kv offload to user space memory (pinned memory)
MMAP:
PWRITE:
'''
EXP_TYPE = os.environ.get("FRIC_EXP", "PURE_INF")

if EXP_TYPE == "PWRITE":
    FILE_NAME = '/mnt/data/xujiahao/beaver/moudels/register_pm/mnt/fric_save.bin'
elif EXP_TYPE == "MMAP":
    FILE_NAME = "/dev/dax0.0"

torch_to_np = {
    torch.float32: np.float32,
    torch.float64: np.float64,
    torch.float16: np.float16,
    torch.int32:   np.int32,
    torch.int64:   np.int64,
    torch.uint8:   np.uint8,
}


class fric_offloader:
    """
    Statically manage asynchronous offloading of key and value tensors to CPU.
    1. Define a copy_stream for asynchronous copy.
    2. Pre-allocate pinned memory buffers for key and value tensors.

    TODO: The current implementation ignores prefill and decode distinction.
    Just allocate the max size of tokens for the Pinned memory buffers.
    
    """
    def __init__(self, max_batch_size:int = 1, max_seq_len:int = 4096, num_heads:int = 8, head_dim:int = 128, num_layers = 32, dtype = torch.float16): # Remember consindering the KVSize of prefill should multli 32 too!
        self.copy_stream = torch.cuda.Stream()
        self.num_layers = num_layers
        self.last_token_id = None
        
        self.offset = [0] * num_layers
        total_size = max_batch_size * num_heads * max_seq_len * head_dim * 2 * num_layers * 4 # bfloat16 = 2 bytes

        if EXP_TYPE == "USER_SPACE" or EXP_TYPE == "PWRITE":
            self.k_buf = [torch.empty((max_batch_size, num_heads, max_seq_len, head_dim), dtype=dtype, device='cpu', pin_memory=True) for _ in range(num_layers)]
            self.v_buf = [torch.empty((max_batch_size, num_heads, max_seq_len, head_dim), dtype=dtype, device='cpu', pin_memory=True) for _ in range(num_layers)]
        
        if EXP_TYPE == "PWRITE" or EXP_TYPE == "MMAP":
            self.fd = os.open(FILE_NAME, os.O_CREAT | os.O_RDWR)
        
        if EXP_TYPE == "PWRITE":
            os.ftruncate(self.fd, total_size)

        if EXP_TYPE == "MMAP":
            self.k_buf = []
            self.k_mmap_handles = [] 
            self.v_buf = []
            self.v_mmap_handles = []

            for i in range(num_layers):
                t, mm = self.__cpu_buffer_mmap(
                    file_size = max_batch_size * num_heads * max_seq_len * head_dim * 2,   # bfloat16
                    shape = (max_batch_size, num_heads, max_seq_len, head_dim),
                    dtype = dtype
                )
                self.k_buf.append(t)
                self.k_mmap_handles.append(mm)

                t, mm = self.__cpu_buffer_mmap(
                    file_size = max_batch_size * num_heads * max_seq_len * head_dim * 2,   # bfloat16
                    shape = (max_batch_size, num_heads, max_seq_len, head_dim),
                    dtype = dtype
                )
                self.v_buf.append(t)
                self.v_mmap_handles.append(mm)
            
        self.test_bytes = torch.randn((max_batch_size, num_heads, 132, head_dim), dtype=dtype, device='cpu').numpy().tobytes()

    def __cpu_buffer_mmap(self, file_size, shape, dtype=torch.float16):
        mm = mmap.mmap(self.fd, file_size, prot=mmap.PROT_READ | mmap.PROT_WRITE)
        np_dtype = torch_to_np[dtype]
        ptr = np.frombuffer(mm, dtype=np_dtype)

        address = ptr.ctypes.data
        size_bytes = ptr.nbytes
        cp.cuda.runtime.hostRegister(address, size_bytes, 0x02)
        t = torch.from_numpy(ptr).view(*shape)
        return t, mm
        

    def write_kv_to_pagecache(self, k_buf, v_buf, offset_k, offset_v):
        k_bytes = k_buf.numpy().tobytes()
        v_bytes = v_buf.numpy().tobytes()
        # k_bytes =  self.test_bytes
        # v_bytes =  self.test_bytes

        os.pwrite(self.fd, k_bytes, offset_k)
        os.pwrite(self.fd, v_bytes, offset_v)

        

    def async_offload(self, layer_idx, k:torch.tensor, v:torch.tensor, event:torch.cuda.Event = None):
        """
        Asynchronously offload key and value tensors to CPU to save GPU memory.
        Args:
            k: Key tensor of shape (B, H, T, D)
            v: Value tensor of shape (B, H, T, D)

        TODO: Now the KV is just save to the pre-allocated buffer.
        1. Implement a ring buffer to manage the KV cache more efficiently.
        2. Save the offloaded KV to disk.

        """
        if k is None or v is None:
            return None, None
        
        B, H, T, D = k.shape
        offset = self.offset[layer_idx]
        assert v.shape == (B, H, T, D)
        assert offset + T <= self.k_buf[layer_idx].shape[2], "FRIC:KVBuffer overflow!"

        with torch.cuda.stream(self.copy_stream):
            self.copy_stream.wait_event(event)
            k_buf = self.k_buf[layer_idx][:B, :H, offset:offset+T, :]
            v_buf = self.v_buf[layer_idx][:B, :H, offset:offset+T, :]
            k_buf = k.to(k_buf, non_blocking=True)
            v_buf = v.to(v_buf, non_blocking=True)
        
        if EXP_TYPE == "PWRITE":
            self.write_kv_to_pagecache(k_buf, v_buf, 0, 0)

        self.offset[layer_idx] += T
        return k_buf, v_buf

    def sync_restore_each_layer(self, layer_idx):
        """
        Restore the offloaded KV to GPU memory. Inference begins after sync restore KV of all layers.
        TODO: 1. Now the implementation is just restore from the DRAM.
              2. Need to implement async restore which makes each layer inference async after load their KV.
        """
        B, H, T, D = self.k_buf[layer_idx].shape 

        k_buf = self.k_buf[layer_idx][:, :, :self.offset[layer_idx], :]
        v_buf = self.v_buf[layer_idx][:, :, :self.offset[layer_idx], :]
        k = k_buf.to('cuda', non_blocking=False)
        v = v_buf.to('cuda', non_blocking=False)
        return k, v


    
    def sync_restore_each_layer_cpu(self, layer_idx, dtype=torch.float32):
        k = self.k_buf[layer_idx][:, :, :self.offset[layer_idx], :].to(dtype=dtype)
        v = self.v_buf[layer_idx][:, :, :self.offset[layer_idx], :].to(dtype=dtype)
        return k, v
    
    def sync_restore(self):
        """
        Return the offloaded KV of all layers.
        """
        past_key_values = tuple(tuple(self.sync_restore_each_layer(layer_id)) for layer_id in range(self.num_layers))

        return past_key_values
    
    def sync_restore_cpu(self, dtype=torch.float32):
        """
        Return the offloaded KV of all layers in CPU.
        """
        past_key_values = tuple(tuple(self.sync_restore_each_layer_cpu(layer_id, dtype=dtype)) for layer_id in range(self.num_layers))

        return past_key_values

    def wait_for_offload(self):
        """
        Ensure all async CPU copies are finished before accessing buffers
        """
        torch.cuda.current_stream().wait_stream(self.copy_stream)

    def show(self):
        """
        For checking the kv buf in experiment
        """
        print(f"FRIC Offset is {self.offset}")

    def set_last_token_id(self, last_token_id):
        """
        TODO: Now the use code is in inference.
        """
        self.last_token_id = last_token_id

