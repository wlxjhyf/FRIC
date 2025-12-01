

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
    

