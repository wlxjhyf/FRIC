import torch
from arch.riscv.io_urings import make_iov
from arch.riscv.index_table import KVIndexTable, IOEntry
from arch.riscv.ssd_blocks import blocks_prepare, log_prepare, BlockBinding
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
    io_uring_peek_cqe,
    io_uring_queue_exit,
)
import threading
import ctypes
import time
import queue

ENTRIES = 64 

import itertools

_id_counter = itertools.count(1)

def generate_unique_id():
    return next(_id_counter)



class fric_offloader:
    """
    Statically manage asynchronous offloading of key and value tensors to CPU.
    1. Define a copy_stream for asynchronous copy.
    2. Pre-allocate pinned memory buffers for key and value tensors.

    TODO: The current implementation ignores prefill and decode distinction.
    Just allocate the max size of tokens for the Pinned memory buffers.
    
    """
    def __init__(self, max_batch_size:int = 1, max_seq_len:int = 1024 * 10, num_heads:int = 8, head_dim:int = 128, num_layers = 32): # Remember consindering the KVSize of prefill should multli 32 too!
        self.copy_stream = torch.cuda.Stream()
        self.num_layers = num_layers
        self.last_token_id = None
        self.k_buf = [torch.empty((max_batch_size, num_heads, max_seq_len, head_dim), device='cpu', pin_memory=True) for _ in range(num_layers)]
        self.v_buf = [torch.empty((max_batch_size, num_heads, max_seq_len, head_dim), device='cpu', pin_memory=True) for _ in range(num_layers)]
        self.offset = [0] * num_layers

        self.log = log_prepare()
        self.block_binding = BlockBinding(blocks_prepare())
        self.kv_index_table = KVIndexTable()
        self.ring = io_uring()
        io_uring_queue_init(ENTRIES, self.ring, 0)
        self.offload_queue = queue.Queue(maxsize=128) 
        self._start_cqe_worker()
        self._start_submit_worker()
        self.io_entries = {}

    def _start_cqe_worker(self):
        t = threading.Thread(target=self._cqe_worker, daemon=True)
        t.start()
    
    def _submit_worker(self):
        while True:
            task = self.offload_queue.get()
            if task is None:
                break   

            k_buf, v_buf, seq, layer, token = task

            try:
                self.submit(
                    k_buf,
                    v_buf,
                    seq=seq,
                    layer=layer,
                    token=token
                )
            except Exception as e:
                print(f"[FRIC submit worker error] {e}")

            self.offload_queue.task_done()

    def _start_submit_worker(self):
        t = threading.Thread(
            target=self._submit_worker,
            daemon=True
        )
        t.start()

    def _cqe_worker(self):
        cqe = io_uring_cqe()

        while True:
            ret = io_uring_peek_cqe(self.ring, cqe)
            if ret < 0:
                time.sleep(0.1)
                continue
            
            io_entry_idx = cqe.user_data
            io_entry = self.io_entries.pop(io_entry_idx, None)

            io_entry.pending -= 1

            if io_entry.pending == 0:
                io_entry.done = True
                self.kv_index_table.add_entry(
                    io_entry.seq,
                    io_entry.kv,
                    io_entry.layer,
                    io_entry.token,
                    io_entry.start_idx,
                    io_entry.end_idx,
                    io_entry.start_offset,
                    io_entry.end_offset,
                )

                if io_entry.end_idx == io_entry.start_idx:
                    self.log.log_partial(
                        io_entry.start_idx,
                        io_entry.start_offset,
                        io_entry.end_offset - io_entry.start_offset
                    )
                else:
                    self.log.log_alloc(
                        io_entry.start_idx,
                        io_entry.end_idx - io_entry.start_idx
                    )
                    self.log.log_flip(
                        io_entry.start_idx,
                        io_entry.end_idx - io_entry.start_idx
                    )

            io_uring_cqe_seen(self.ring, cqe)

    def _write_iov(self, iov, start_idx, start_offset, io_entry):
        base = iov.iov_base
        length = iov.iov_len

        work_idx = start_idx
        work_offset = start_offset
        written = 0

        if work_idx == 0 and work_offset == 0:
            self.block_binding.alloc()

        while written < length:
            sqe = io_uring_get_sqe(self.ring)

            write_len = min(
                self.kv_index_table.capacity - work_offset,
                length - written
            )

            chunk = base[written: written + write_len]
            fd = self.block_binding.fd_of(work_idx)

            io_uring_prep_write(
                sqe, fd, chunk, write_len, work_offset
            )

            io_entry_idx = generate_unique_id()
            self.io_entries[io_entry_idx] = io_entry
            sqe.user_data = io_entry_idx

            io_entry.pending += 1

            written += write_len
            work_offset += write_len

            if work_offset == self.kv_index_table.capacity:
                work_idx += 1
                self.block_binding.alloc()
                work_offset = 0

        io_uring_submit(self.ring)
        return work_idx, work_offset

    def submit(self, k_buf, v_buf, seq, layer, token):
        k_iov = make_iov(k_buf)
        v_iov = make_iov(v_buf)

        for kv, iov in (("K", k_iov), ("V", v_iov)):
            start_idx = self.kv_index_table.idx
            start_offset = self.kv_index_table.offset

            io_entry = IOEntry(
                seq=seq,
                kv=kv,
                layer=layer,
                token=token,
                start_idx=start_idx,
                start_offset=start_offset,
            )

            end_idx, end_offset = self._write_iov(
                iov,
                start_idx,
                start_offset,
                io_entry
            )

            io_entry.end_idx = end_idx
            io_entry.end_offset = end_offset

            self.kv_index_table.add_entry(
                seq, kv, layer, token,
                start_idx, end_idx,
                start_offset, end_offset
            )

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

        self.offset[layer_idx] += T
        # k_iov = self.buffer_slice(k_buf, B, H, T)
        # v_iov = self.buffer_slice(v_buf, B, H, T)

        # self.submit(k_buf, v_buf, seq=0, layer=layer_idx, token=T)
        self.offload_queue.put(
            (k_buf, v_buf, 0, layer_idx, T)
        )
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
    
    def sync_restore(self):
        """
        Return the offloaded KV of all layers.
        """
        past_key_values = tuple(tuple(self.sync_restore_each_layer(layer_id)) for layer_id in range(self.num_layers))

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