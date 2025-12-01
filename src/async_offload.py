import torch
from arch.riscv.io_urings import __make_iov, submit
from arch.riscv.ssd_blocks import blocks_prepare

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
        self.fds = blocks_prepare()

    # def buffer_slice(self, tensor, batch, head, tokens):
    #     base_ptr = tensor.data_ptr()
    #     elem_size = tensor.element_size()
        
    #     offset_elems = (
    #         batch * self.num_heads * self.max_seq_len * self.head_dim +
    #         head  * self.max_seq_len * self.head_dim
    #     )

    #     offset_bytes = offset_elems * elem_size
    #     write_ptr = base_ptr + offset_bytes

    #     length = tokens * self.head_dim * elem_size
    #     iov = __make_iov(write_ptr, length)
    #     return iov

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

        submit(self.fds, k_buf, v_buf, seq=0, layer=layer_idx, token=T)
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