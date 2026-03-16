import torch


class QueryWindowBuffer:
    def __init__(self, num_layers, window_size, num_heads, head_dim, device, dtype):
        self.data = torch.empty(
            num_layers, window_size, num_heads, head_dim,
            device=device, dtype=dtype
        )
        self.write_pos = 0
        self.valid_len = 0
        self.active = True

    def reset(self):
        self.write_pos = 0
        self.valid_len = 0
        self.active = False


class QueryWindowManager:

    def __init__(self, num_layers, window_size, num_heads, head_dim, device="cuda", dtype=torch.float16):
        self.num_layers = num_layers
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype

        self.buffers: dict[int, QueryWindowBuffer] = {}

    def has(self, seq_id: int) -> bool:
        return seq_id in self.buffers

    def activate(self, seq_id: int):
        if seq_id not in self.buffers:
            self.buffers[seq_id] = QueryWindowBuffer(
                self.num_layers, self.window_size, self.num_heads, self.head_dim,
                self.device, self.dtype
            )
        else:
            buf = self.buffers[seq_id]
            buf.write_pos = 0
            buf.valid_len = 0
            buf.active = True

    def append(self, seq_ids: list[int], layer_id: int, q: torch.Tensor):

        assert q.dim() == 3
        assert len(seq_ids) == q.size(0)
        for i, seq_id in enumerate(seq_ids):
            buf = self.buffers[seq_id]
            pos = buf.write_pos
            buf.data[layer_id, pos].copy_(q[i])
        if layer_id == self.num_layers - 1:
            for seq_id in seq_ids:
                buf = self.buffers[seq_id]
                buf.write_pos = (buf.write_pos + 1) % self.window_size
                if buf.valid_len < self.window_size:
                    buf.valid_len += 1

    def gather(self, seq_ids: list[int], layer_id: int) -> torch.Tensor:

        outputs = []
        for seq_id in seq_ids:
            assert seq_id in self.buffers, f"seq_id={seq_id} not found in query window buffers"
            buf = self.buffers[seq_id]

            pos = buf.write_pos
            layer_buf = buf.data[layer_id]  # [W, H, D]

            if pos == 0:
                q_hist = layer_buf
            else:
                q_hist = torch.cat([layer_buf[pos:], layer_buf[:pos]], dim=0)

            outputs.append(q_hist)

        q_batch = torch.stack(outputs, dim=0)                  # [bsz, W, H, D]
        q_batch = q_batch.permute(0, 2, 1, 3).contiguous()    # [bsz, H, W, D]
        return q_batch

    def free(self, seq_id: int):
        if seq_id in self.buffers:
            del self.buffers[seq_id]