
from dataclasses import dataclass
from typing import List, Tuple
import threading


class KVIndexTable:
    def __init__(self):
        # 多级嵌套字典
        # table[seq][KV][layer][token] = {"block_id": ..., "offset": ..., "length": ...}
        self.capacity = 2 * 1024 * 1024
        self.table = {}
        # 先支持单个对话
        self.idx = 0 
        self.offset = 0

    '''
    kv: 0 means k, 1 means v
    '''
    def add_entry(self, seq, kv, layer, token, start_idx, end_idx, start_offset, end_offset):
        self.table.setdefault(seq, {}) \
                  .setdefault(kv, {}) \
                  .setdefault(layer, {})[token] = {
                      "start_idx": start_idx,
                      "end_idx": end_idx,
                      "start_offset": start_offset,
                      "end_offset": end_offset
                  }
        self.idx = end_idx
        self.offset = end_offset

    def query(self, seq, kv, layer, token):
        """返回单条索引记录，如果不存在返回 None"""
        return self.table.get(seq, {}) \
                         .get(kv, {}) \
                         .get(layer, {}) \
                         .get(token, None)
    

class IOEntry:
    __slots__ = (
        "seq", "kv", "layer", "token",
        "start_idx", "start_offset",
        "end_idx", "end_offset",
        "pending",
        "done",
    )

    def __init__(
        self,
        seq, kv, layer, token,
        start_idx, start_offset,
    ):
        self.seq = seq
        self.kv = kv
        self.layer = layer
        self.token = token

        self.start_idx = start_idx
        self.start_offset = start_offset
        self.end_idx = None
        self.end_offset = None

        self.pending = 0
        self.done = False




@dataclass(frozen=True)
class FRIC_SSD_LogEntry:
    """
    Log entry

    W[m, n] : allocate and write blocks [m, m+n)
    P[m, s, l] : partial append to block m
    F[m, n] : free blocks [m, m+n)
    """
    op: str
    args: Tuple


class FRIC_SSD_Log:
    def __init__(self):
        self.entries: List[FRIC_SSD_LogEntry] = []
        self.tail: int = 0          # log tail (LSN)
        self._lock = threading.Lock()

    # ---------- 核心原子提交点 ----------

    def _append_and_commit(self, entry: FRIC_SSD_LogEntry) -> int:
        """
        原子地：
        1. append log entry
        2. tail += 1

        返回该 entry 的 LSN
        """
        with self._lock:
            lsn = self.tail
            self.entries.append(entry)
            self.tail += 1
            return lsn

    # ---------- Log operations ----------

    def log_write(self, start: int, count: int) -> int:
        """
        W[m, n]
        """
        entry = FRIC_SSD_LogEntry('W', (start, count))
        return self._append_and_commit(entry)

    def log_partial(self, block: int, offset: int, length: int) -> int:
        """
        P[m, s, l]
        """
        entry = FRIC_SSD_LogEntry('P', (block, offset, length))
        return self._append_and_commit(entry)

    def log_free(self, start: int, count: int) -> int:
        """
        F[m, n]
        """
        entry = FRIC_SSD_LogEntry('F', (start, count))
        return self._append_and_commit(entry)

    # ---------- Recovery ----------

    def replay(self):
        """
        崩溃恢复：
        tail = 已成功 replay 的 log 数
        """
        applied = 0
        for entry in self.entries:
            # 这里只是示意：真正的 apply 逻辑你可以接 SSD / KV
            applied += 1
        self.tail = applied

    # ---------- Debug ----------

    def __repr__(self):
        return "\n".join(
            f"[{i}] {e.op}{e.args}"
            for i, e in enumerate(self.entries)
        )
